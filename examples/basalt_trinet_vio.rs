//! Stereo-inertial VIO over a whole Trinet Pro Stereo take, streamed: both eyes are decoded
//! straight from the camera's MP4s through ffmpeg (no images on disk), the IMU is walked with
//! a cursor, and the lenses are the factory Kannala-Brandt (`kb4`) model, so frames are only
//! scaled, never resampled.
//!
//! The take is described by a small manifest directory (written by sx's ego editor,
//! `examples/vio_export.rs`):
//!
//! ```text
//! calibration.json   Basalt calibration: kb4 cameras, T_imu_cam, the factory IMU model
//! frames.csv         left_frame,right_frame,timestamp_ns   (camera time on the IMU clock)
//! imu.csv            EuRoC imu0 columns: timestamp_ns, gyro xyz (rad/s), accel xyz (m/s²)
//! video.txt          left=<mp4>  right=<mp4>  fps=<f>  width=<w>  height=<h>
//! ```
//!
//! ```text
//! cargo run --release --example basalt_trinet_vio -- \
//!   --manifest /tmp/take0014-vio --config configs/basalt/euroc_config.json --out-dir out
//! ```
//!
//! Writes `trajectory.tum` (IMU pose in the VIO world, gravity along −z), `tracking.csv`
//! (per frame: observations in each eye, IMU samples; few observations means the tracker is
//! losing the scene) and `summary.txt`. With `--map`, also `map.csv`: every landmark the
//! window held, at its last optimized world position (a sparse map for viewers).

use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdout, Command, Stdio},
    time::Instant,
};

use nalgebra::Vector3;
use visloc_basalt::{
    config::BasaltConfig, types::ImuSample, BasaltCalibration, BasaltVioEstimatorAdapter,
    EurocSensorFrame, RawU16Image, RollingShutter,
};

struct Video {
    left: PathBuf,
    right: PathBuf,
    fps: f64,
    width: usize,
    height: usize,
}

fn read_video(path: &PathBuf) -> Result<Video, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let kv: BTreeMap<&str, &str> = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();
    let get = |k: &str| kv.get(k).copied().ok_or(format!("video.txt: missing {k}"));
    Ok(Video {
        left: PathBuf::from(get("left")?),
        right: PathBuf::from(get("right")?),
        fps: get("fps")?.parse()?,
        width: get("width")?.parse()?,
        height: get("height")?.parse()?,
    })
}

/// One eye decoded to 8-bit grey at the manifest's size, from frame `first` onward.
struct Decoder {
    child: Child,
    out: ChildStdout,
    buf: Vec<u8>,
    width: usize,
    height: usize,
}

impl Decoder {
    fn open(
        video: &PathBuf,
        fps: f64,
        first: usize,
        count: usize,
        width: usize,
        height: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Seek half a frame early: the first decoded frame is then exactly `first`.
        let start = (first as f64 - 0.5).max(0.0) / fps;
        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-ss",
                &format!("{start:.6}"),
                "-i",
            ])
            .arg(video)
            .args(["-frames:v", &count.to_string()])
            .args([
                "-vf",
                &format!("scale={width}:{height}:flags=area,format=gray"),
            ])
            // Every decoded frame exactly once: the MP4 declares 30 fps but runs at ~29.94, and
            // a constant-rate output would duplicate a frame every ~500, sliding the video
            // against its timestamps.
            .args(["-fps_mode", "passthrough"])
            .args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let out = child.stdout.take().ok_or("ffmpeg stdout")?;
        Ok(Self {
            child,
            out,
            buf: vec![0; width * height],
            width,
            height,
        })
    }

    fn next(&mut self) -> Result<RawU16Image, Box<dyn std::error::Error>> {
        self.out.read_exact(&mut self.buf)?;
        // Upstream's 8-bit conversion: u16 = u8 << 8.
        let pixels = self.buf.iter().map(|v| u16::from(*v) << 8).collect();
        Ok(RawU16Image::new(self.width, self.height, pixels)?)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = None;
    let mut config = PathBuf::from("configs/basalt/euroc_config.json");
    let mut out_dir = PathBuf::from("target/basalt_trinet");
    let mut max_frames: Option<usize> = None;
    let mut skip_frames = 0usize;
    let mut rolling_shutter = true;
    let mut map = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(next()?)),
            "--config" => config = PathBuf::from(next()?),
            "--out-dir" => out_dir = PathBuf::from(next()?),
            "--max-frames" => max_frames = Some(next()?.parse()?),
            "--skip-frames" => skip_frames = next()?.parse()?,
            "--no-rolling-shutter" => rolling_shutter = false,
            "--map" => map = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let manifest = manifest.ok_or("--manifest DIR is required")?;
    let calibration = BasaltCalibration::from_path(manifest.join("calibration.json"))?;
    let config = BasaltConfig::from_json(&fs::read_to_string(&config)?)?;
    let video = read_video(&manifest.join("video.txt"))?;

    // left_frame,right_frame,timestamp_ns[,readout_ns,centered]
    let mut frames: Vec<(usize, usize, i64, Option<RollingShutter>)> = Vec::new();
    for line in fs::read_to_string(manifest.join("frames.csv"))?
        .lines()
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 3 {
            return Err("frames.csv: expected at least 3 columns".into());
        }
        let rs = match (f.get(3), f.get(4)) {
            (Some(r), Some(c)) if rolling_shutter => {
                let readout_ns: i64 = r.parse()?;
                (readout_ns > 0).then_some(RollingShutter {
                    readout_ns,
                    centered: *c == "1",
                })
            }
            _ => None,
        };
        frames.push((f[0].parse()?, f[1].parse()?, f[2].parse()?, rs));
    }
    frames.drain(..skip_frames.min(frames.len()));
    if let Some(n) = max_frames {
        frames.truncate(n);
    }
    if frames.is_empty() {
        return Err("frames.csv lists no frames".into());
    }
    let mut imu: Vec<ImuSample> = Vec::new();
    for line in fs::read_to_string(manifest.join("imu.csv"))?.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<Result<_, _>>()?;
        imu.push(ImuSample::new(
            v[0] as i64,
            Vector3::new(v[1], v[2], v[3]),
            Vector3::new(v[4], v[5], v[6]),
        ));
    }

    let (first_l, first_r, _, _) = frames[0];
    let n = frames.len();
    let mut left = Decoder::open(
        &video.left,
        video.fps,
        first_l,
        n,
        video.width,
        video.height,
    )?;
    let mut right = Decoder::open(
        &video.right,
        video.fps,
        first_r,
        n,
        video.width,
        video.height,
    )?;

    let mut adapter = BasaltVioEstimatorAdapter::from_config(&calibration, &config)?;
    adapter.estimator.set_lean_marg_data(true);
    fs::create_dir_all(&out_dir)?;
    let mut tum = BufWriter::new(fs::File::create(out_dir.join("trajectory.tum"))?);
    let mut tracking = BufWriter::new(fs::File::create(out_dir.join("tracking.csv"))?);
    writeln!(
        tracking,
        "left_frame,timestamp_ns,cam0_observations,cam1_observations,imu_samples"
    )?;

    let started = Instant::now();
    // Start the IMU 0.2 s before the first frame, as a recording that began there would.
    let lead_in = frames[0].2 - 200_000_000;
    let mut cursor = imu.partition_point(|s| s.timestamp_ns < lead_in);
    let mut previous: Option<i64> = None;
    let mut processed = 0usize;
    let mut weak = 0usize;
    // Landmark id -> (last optimized world position, frames it was in the window).
    let mut landmarks: BTreeMap<u64, ([f64; 3], u32)> = BTreeMap::new();
    for (index, (lf, _rf, t, rs)) in frames.iter().copied().enumerate() {
        adapter.set_rolling_shutter(rs);
        let cam0 = left.next()?;
        let cam1 = right.next()?;
        // IMU after the previous frame, through this one (frame zero: everything up to it).
        let begin = cursor;
        while cursor < imu.len() && imu[cursor].timestamp_ns <= t {
            cursor += 1;
        }
        let slice: Vec<ImuSample> = imu[begin..cursor]
            .iter()
            .filter(|s| previous.is_none_or(|p| s.timestamp_ns > p))
            .copied()
            .collect();
        let frame = EurocSensorFrame {
            frame_id: index as u64,
            timestamp_ns: t,
            cam0,
            cam1: Some(cam1),
            cam0_path: video.left.clone(),
            cam1_path: Some(video.right.clone()),
            imu: slice,
            initialization_imu: (index == 0).then(|| imu.get(cursor).copied()).flatten(),
        };
        previous = Some(t);
        let output = adapter.process_without_marg_data_no_trace(frame)?;
        let pose = &output.estimator.state.imu_to_world;
        let q = pose.rotation.quaternion();
        writeln!(
            tum,
            "{:.9} {:.9} {:.9} {:.9} {:.12} {:.12} {:.12} {:.12}",
            t as f64 * 1e-9,
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            q.i,
            q.j,
            q.k,
            q.w
        )?;
        let c0 = output
            .tracks
            .observations
            .iter()
            .filter(|o| o.camera_id == 0)
            .count();
        let c1 = output
            .tracks
            .observations
            .iter()
            .filter(|o| o.camera_id == 1)
            .count();
        if c0 < 30 {
            weak += 1;
        }
        writeln!(tracking, "{lf},{t},{c0},{c1},{}", output.imu_count)?;
        if map {
            for (id, p) in adapter.estimator.landmark_positions() {
                let e = landmarks.entry(id).or_insert(([0.0; 3], 0));
                *e = ([p.x, p.y, p.z], e.1 + 1);
            }
        }
        processed += 1;
        if processed % 900 == 0 {
            let secs = started.elapsed().as_secs_f64();
            eprintln!(
                "{processed}/{n} frames, {:.2}x real time",
                processed as f64 / video.fps / secs
            );
        }
    }
    tum.flush()?;
    tracking.flush()?;
    if map {
        let mut out = BufWriter::new(fs::File::create(out_dir.join("map.csv"))?);
        writeln!(out, "landmark,x,y,z,frames")?;
        for (id, ([x, y, z], frames)) in &landmarks {
            writeln!(out, "{id},{x:.5},{y:.5},{z:.5},{frames}")?;
        }
        out.flush()?;
    }
    let secs = started.elapsed().as_secs_f64();
    let summary = format!(
        "frames={processed}\nseconds_of_video={:.1}\nwall_seconds={secs:.1}\nrealtime_factor={:.3}\nweak_frames={weak}\nimu_samples={}\n",
        processed as f64 / video.fps,
        processed as f64 / video.fps / secs,
        imu.len()
    );
    fs::write(out_dir.join("summary.txt"), &summary)?;
    print!("{summary}");
    Ok(())
}
