//! Deterministic Basalt streaming and minimal VIO window contracts.
use std::collections::VecDeque;

use crate::imu::TimestampedImu;

pub const OF_INPUT_CAPACITY: usize = 10;
pub const VISION_INPUT_CAPACITY: usize = 10;
/// Upstream holds 300 samples (1.5 s of EuRoC's 200 Hz). Trinet runs at ~400 Hz, and a take
/// starts with a short IMU lead-in before its first frame, so hold 3 s at 400 Hz. Capacity is
/// backpressure only; it never changes the arithmetic.
pub const IMU_INPUT_CAPACITY: usize = 1200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionFrame {
    pub frame_id: u64,
    pub timestamp_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMessage {
    Vision(VisionFrame),
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    OfInputFull,
    VisionInputFull,
    ImuInputFull,
    AlreadyEnded,
    NonMonotonicVision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowPhase {
    ImuPredict,
    TrackAssociation,
    ConnectedRatioAndKeyframeDecision,
    TriangulationCandidate,
    LostLandmarkRemoval,
    OptimizeHook,
    MarginalizeHook,
    Output,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VioWindow {
    pub frame_ids: Vec<u64>,
    pub landmark_ids: Vec<u64>,
}

impl VioWindow {
    pub fn process(&mut self, frame: VisionFrame, phases: &mut Vec<WindowPhase>) {
        phases.extend([
            WindowPhase::ImuPredict,
            WindowPhase::TrackAssociation,
            WindowPhase::ConnectedRatioAndKeyframeDecision,
            WindowPhase::TriangulationCandidate,
            WindowPhase::LostLandmarkRemoval,
            WindowPhase::OptimizeHook,
            WindowPhase::MarginalizeHook,
            WindowPhase::Output,
        ]);
        self.frame_ids.push(frame.frame_id);
    }
}

#[derive(Debug)]
pub struct BasaltStream {
    pub enforce_realtime: bool,
    of_input: VecDeque<u64>,
    vision_input: VecDeque<VisionFrame>,
    imu_input: VecDeque<TimestampedImu>,
    ended: bool,
    last_vision_ns: Option<i64>,
    pub window: VioWindow,
    pub phase_log: Vec<WindowPhase>,
}

impl BasaltStream {
    pub fn new(enforce_realtime: bool) -> Self {
        Self {
            enforce_realtime,
            of_input: VecDeque::new(),
            vision_input: VecDeque::new(),
            imu_input: VecDeque::new(),
            ended: false,
            last_vision_ns: None,
            window: VioWindow::default(),
            phase_log: Vec::new(),
        }
    }
    pub fn push_of_input(&mut self, id: u64) -> Result<(), StreamError> {
        if self.of_input.len() >= OF_INPUT_CAPACITY {
            return Err(StreamError::OfInputFull);
        }
        self.of_input.push_back(id);
        Ok(())
    }
    pub fn push_imu(&mut self, sample: TimestampedImu) -> Result<(), StreamError> {
        if self.ended {
            return Err(StreamError::AlreadyEnded);
        }
        if self.imu_input.len() >= IMU_INPUT_CAPACITY {
            return Err(StreamError::ImuInputFull);
        }
        self.imu_input.push_back(sample);
        Ok(())
    }
    pub fn push_vision(&mut self, frame: VisionFrame) -> Result<(), StreamError> {
        if self.ended {
            return Err(StreamError::AlreadyEnded);
        }
        if self.last_vision_ns.is_some_and(|t| frame.timestamp_ns <= t) {
            return Err(StreamError::NonMonotonicVision);
        }
        if self.vision_input.len() >= VISION_INPUT_CAPACITY {
            return Err(StreamError::VisionInputFull);
        }
        self.last_vision_ns = Some(frame.timestamp_ns);
        self.vision_input.push_back(frame);
        Ok(())
    }
    pub fn push(&mut self, message: StreamMessage) -> Result<(), StreamError> {
        match message {
            StreamMessage::Vision(f) => self.push_vision(f),
            StreamMessage::End => self.finish(),
        }
    }
    pub const fn finish(&mut self) -> Result<(), StreamError> {
        if self.ended {
            return Err(StreamError::AlreadyEnded);
        }
        self.ended = true;
        Ok(())
    }
    pub const fn is_ended(&self) -> bool {
        self.ended
    }
    pub fn process_pending(&mut self) -> usize {
        let mut count = 0;
        while let Some(frame) = self.vision_input.pop_front() {
            self.window.process(frame, &mut self.phase_log);
            count += 1;
            if self.enforce_realtime {
                break;
            }
        }
        count
    }
    pub fn pending_vision(&self) -> usize {
        self.vision_input.len()
    }
    pub fn pending_imu(&self) -> usize {
        self.imu_input.len()
    }

    /// Removes IMU samples consumed by the current estimator window.
    ///
    /// The fixed input capacity mirrors Basalt's bounded producer queue.  The
    /// estimator owns the actual preintegration boundary, so this small
    /// adapter lets it acknowledge consumption after a vision event without
    /// exposing the queue itself.
    pub fn drain_pending_imu(&mut self) -> usize {
        let count = self.imu_input.len();
        self.imu_input.clear();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    fn imu(t: i64) -> TimestampedImu {
        crate::ImuSample::new(t, Vector3::zeros(), Vector3::zeros())
    }
    fn frame(i: u64) -> VisionFrame {
        VisionFrame {
            frame_id: i,
            timestamp_ns: i as i64 + 1,
        }
    }

    #[test]
    fn capacities_and_end_sentinel_are_fixed() {
        let mut s = BasaltStream::new(false);
        for i in 0..OF_INPUT_CAPACITY as u64 {
            s.push_of_input(i).unwrap();
        }
        assert_eq!(s.push_of_input(99), Err(StreamError::OfInputFull));
        for i in 0..VISION_INPUT_CAPACITY as u64 {
            s.push_vision(frame(i)).unwrap();
        }
        assert_eq!(s.push_vision(frame(100)), Err(StreamError::VisionInputFull));
        for i in 0..IMU_INPUT_CAPACITY {
            s.push_imu(imu(i as i64)).unwrap();
        }
        assert_eq!(s.push_imu(imu(999)), Err(StreamError::ImuInputFull));
        s.finish().unwrap();
        assert!(s.is_ended());
        assert_eq!(s.push_vision(frame(101)), Err(StreamError::AlreadyEnded));
    }

    #[test]
    fn realtime_false_processes_every_queued_vision_frame_without_drop() {
        let mut s = BasaltStream::new(false);
        for i in 0..VISION_INPUT_CAPACITY as u64 {
            s.push_vision(frame(i)).unwrap();
        }
        assert_eq!(s.process_pending(), VISION_INPUT_CAPACITY);
        assert_eq!(s.pending_vision(), 0);
        assert_eq!(s.window.frame_ids.len(), VISION_INPUT_CAPACITY);
    }

    #[test]
    fn each_frame_has_the_basalt_measure_order() {
        let mut s = BasaltStream::new(false);
        s.push_vision(frame(0)).unwrap();
        s.process_pending();
        assert_eq!(
            s.phase_log,
            vec![
                WindowPhase::ImuPredict,
                WindowPhase::TrackAssociation,
                WindowPhase::ConnectedRatioAndKeyframeDecision,
                WindowPhase::TriangulationCandidate,
                WindowPhase::LostLandmarkRemoval,
                WindowPhase::OptimizeHook,
                WindowPhase::MarginalizeHook,
                WindowPhase::Output
            ]
        );
    }
}
