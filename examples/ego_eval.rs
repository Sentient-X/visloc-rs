//! Egocentric trajectory evaluation the way Human Archive's HA-SLAM report scores systems
//! (ha-slam.humanarchive.ai, July 2026): three numbers per segment, because no one number says
//! whether a SLAM system works, plus how hard the segment's head motion was.
//!
//! - **Completeness**: the share of reference frames that received a pose.
//! - **Raw ATE**: position error after a rigid (SE(3)) alignment, over the frames that have a
//!   pose. Reported as unusable ("X") when completeness is below 85 %, since an error over a
//!   self-selected slice says which frames were accepted, not how well the run tracked.
//! - **Gap-adjusted ATE**: every reference frame scored; where there is no pose, the last pose
//!   is held, as a runtime would show it, under the same alignment.
//! - **Difficulty**: mean and peak head rotation (deg/s) and mean speed (m/s) from the reference,
//!   bucketed easy / medium / hard by mean rotation.
//!
//! ```text
//! cargo run --release --example ego_eval -- ESTIMATE.tum REFERENCE.tum \
//!   [--segment-seconds 60] [--tolerance-ms 10] [--json out.json]
//! ```
//! Both files are TUM (`t tx ty tz qx qy qz qw`, t in seconds) on the same clock.

use std::{fs, path::PathBuf};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::umeyama_similarity_transform;

#[derive(Clone, Copy, Debug)]
struct Pose {
    t: f64,
    p: Vector3<f64>,
    q: UnitQuaternion<f64>,
}

fn read_tum(path: &PathBuf) -> Result<Vec<Pose>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = l
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if v.len() < 8 {
            return Err(format!("{}: expected 8 columns", path.display()).into());
        }
        out.push(Pose {
            t: v[0],
            p: Vector3::new(v[1], v[2], v[3]),
            q: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[7], v[4], v[5], v[6])),
        });
    }
    out.sort_by(|a, b| a.t.total_cmp(&b.t));
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
struct Scores {
    frames: usize,
    completeness: f64,
    raw_ate_rmse_m: Option<f64>,
    raw_ate_mean_m: Option<f64>,
    gap_adjusted_ate_rmse_m: f64,
    mean_rotation_deg_s: f64,
    peak_rotation_deg_s: f64,
    mean_speed_m_s: f64,
    bucket: &'static str,
}

/// The estimated pose at reference time `t`, if one lies within `tol` seconds.
fn matched(est: &[Pose], t: f64, tol: f64) -> Option<Pose> {
    let i = est.partition_point(|e| e.t < t);
    [i.checked_sub(1), Some(i)]
        .into_iter()
        .flatten()
        .filter_map(|j| est.get(j))
        .filter(|e| (e.t - t).abs() <= tol)
        .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
        .copied()
}

/// The newest estimated pose at or before `t` (the frozen pose a runtime would still show);
/// before the first pose, the first one.
fn held(est: &[Pose], t: f64, tol: f64) -> Option<Pose> {
    let i = est.partition_point(|e| e.t <= t + tol);
    est.get(i.saturating_sub(1)).copied()
}

fn difficulty(reference: &[Pose]) -> (f64, f64, f64) {
    if reference.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let dur = reference[reference.len() - 1].t - reference[0].t;
    let mut angle = 0.0;
    let mut dist = 0.0;
    for w in reference.windows(2) {
        angle += (w[0].q.inverse() * w[1].q).angle();
        dist += (w[1].p - w[0].p).norm();
    }
    // Peak: the fastest 0.1 s of rotation.
    let mut peak: f64 = 0.0;
    let mut j = 0;
    for i in 0..reference.len() {
        while j < reference.len() && reference[j].t - reference[i].t < 0.1 {
            j += 1;
        }
        if j >= reference.len() {
            break;
        }
        let dt = reference[j].t - reference[i].t;
        if dt > 0.0 {
            peak = peak.max((reference[i].q.inverse() * reference[j].q).angle() / dt);
        }
    }
    (
        angle.to_degrees() / dur.max(1e-9),
        peak.to_degrees(),
        dist / dur.max(1e-9),
    )
}

fn score(est: &[Pose], reference: &[Pose], tol: f64) -> Scores {
    let pairs: Vec<(Pose, Pose)> = reference
        .iter()
        .filter_map(|r| matched(est, r.t, tol).map(|e| (e, *r)))
        .collect();
    let completeness = pairs.len() as f64 / reference.len().max(1) as f64;
    let src: Vec<Point3<f64>> = pairs.iter().map(|(e, _)| Point3::from(e.p)).collect();
    let dst: Vec<Point3<f64>> = pairs.iter().map(|(_, r)| Point3::from(r.p)).collect();
    let align = umeyama_similarity_transform(&src, &dst, false);
    let apply = |p: Vector3<f64>| match &align {
        Some(a) => a.apply(&Point3::from(p)).coords,
        None => p,
    };
    let raw: Vec<f64> = pairs
        .iter()
        .map(|(e, r)| (apply(e.p) - r.p).norm())
        .collect();
    let rmse = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt();
    let gap: Vec<f64> = reference
        .iter()
        .filter_map(|r| held(est, r.t, tol).map(|e| (apply(e.p) - r.p).norm()))
        .collect();
    let usable = completeness >= 0.85 && !raw.is_empty();
    let (mean_rot, peak_rot, speed) = difficulty(reference);
    Scores {
        frames: reference.len(),
        completeness,
        raw_ate_rmse_m: usable.then(|| rmse(&raw)),
        raw_ate_mean_m: usable.then(|| raw.iter().sum::<f64>() / raw.len() as f64),
        gap_adjusted_ate_rmse_m: if gap.is_empty() { f64::NAN } else { rmse(&gap) },
        mean_rotation_deg_s: mean_rot,
        peak_rotation_deg_s: peak_rot,
        mean_speed_m_s: speed,
        // HA-SLAM's buckets: easy 26–36, medium 41–50, hard 72 deg/s mean rotation.
        bucket: if mean_rot < 40.0 {
            "easy"
        } else if mean_rot < 60.0 {
            "medium"
        } else {
            "hard"
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut segment: Option<f64> = None;
    let mut tol = 0.010;
    let mut json_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--segment-seconds" => segment = Some(next()?.parse()?),
            "--tolerance-ms" => tol = next()?.parse::<f64>()? / 1000.0,
            "--json" => json_out = Some(PathBuf::from(next()?)),
            _ => positional.push(PathBuf::from(a)),
        }
    }
    let [est_path, ref_path] = positional.as_slice() else {
        return Err("usage: ego_eval ESTIMATE.tum REFERENCE.tum [--segment-seconds S] [--tolerance-ms MS] [--json FILE]".into());
    };
    let est = read_tum(est_path)?;
    let reference = read_tum(ref_path)?;
    if reference.is_empty() {
        return Err("empty reference".into());
    }
    let span = segment.unwrap_or(f64::INFINITY);
    let mut segments: Vec<Vec<Pose>> = Vec::new();
    for r in &reference {
        match segments.last_mut() {
            Some(s) if r.t - s[0].t < span => s.push(*r),
            _ => segments.push(vec![*r]),
        }
    }
    let fmt = |v: Option<f64>| v.map_or("X".to_owned(), |x| format!("{:.2} cm", x * 100.0));
    println!("segment  bucket  frames  completeness  raw ATE    gap-adj ATE  rot mean/peak deg/s  speed m/s");
    let mut rows = Vec::new();
    for (k, seg) in segments.iter().enumerate() {
        // Each segment is aligned on its own, as each of HA-SLAM's segments is a separate run.
        let lo = seg[0].t - 1.0;
        let hi = seg[seg.len() - 1].t + 1.0;
        let est_seg: Vec<Pose> = est
            .iter()
            .filter(|e| e.t >= lo && e.t <= hi)
            .copied()
            .collect();
        let s = score(&est_seg, seg, tol);
        println!(
            "{k:>7}  {:<6}  {:>6}  {:>11.1}%  {:>9}  {:>11}  {:>8.1} / {:<8.1}  {:>8.2}",
            s.bucket,
            s.frames,
            s.completeness * 100.0,
            fmt(s.raw_ate_rmse_m),
            fmt(Some(s.gap_adjusted_ate_rmse_m)),
            s.mean_rotation_deg_s,
            s.peak_rotation_deg_s,
            s.mean_speed_m_s
        );
        rows.push(s);
    }
    let n = rows.len().max(1) as f64;
    let mean = |f: &dyn Fn(&Scores) -> f64| rows.iter().map(f).sum::<f64>() / n;
    println!(
        "   mean          {:>6}  {:>11.1}%  {:>9}  {:>11}",
        rows.iter().map(|r| r.frames).sum::<usize>(),
        mean(&|r| r.completeness) * 100.0,
        fmt(rows
            .iter()
            .all(|r| r.raw_ate_rmse_m.is_some())
            .then(|| mean(&|r| r.raw_ate_rmse_m.unwrap_or(0.0)))),
        fmt(Some(mean(&|r| r.gap_adjusted_ate_rmse_m)))
    );
    if let Some(path) = json_out {
        let body: Vec<String> = rows
            .iter()
            .enumerate()
            .map(|(k, r)| {
                format!(
                    "{{\"segment\":{k},\"bucket\":\"{}\",\"frames\":{},\"completeness\":{},\"raw_ate_rmse_m\":{},\"raw_ate_mean_m\":{},\"gap_adjusted_ate_rmse_m\":{},\"mean_rotation_deg_s\":{},\"peak_rotation_deg_s\":{},\"mean_speed_m_s\":{}}}",
                    r.bucket,
                    r.frames,
                    r.completeness,
                    r.raw_ate_rmse_m.map_or("null".into(), |v| v.to_string()),
                    r.raw_ate_mean_m.map_or("null".into(), |v| v.to_string()),
                    r.gap_adjusted_ate_rmse_m,
                    r.mean_rotation_deg_s,
                    r.peak_rotation_deg_s,
                    r.mean_speed_m_s
                )
            })
            .collect();
        fs::write(path, format!("[\n{}\n]\n", body.join(",\n")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head turning at `deg_s` about z while walking at 0.3 m/s, at 30 Hz for `secs`.
    fn path(secs: f64, deg_s: f64) -> Vec<Pose> {
        (0..(secs * 30.0) as usize)
            .map(|i| {
                let t = i as f64 / 30.0;
                Pose {
                    t,
                    p: Vector3::new(0.3 * t, 0.1 * (t * 2.0).sin(), 1.6),
                    q: UnitQuaternion::from_euler_angles(0.0, 0.0, (deg_s * t).to_radians()),
                }
            })
            .collect()
    }

    #[test]
    fn a_perfect_run_in_another_frame_scores_zero() {
        let r = path(20.0, 30.0);
        // The estimate lives in its own world frame: rotated and shifted.
        let g = UnitQuaternion::from_euler_angles(0.1, -0.2, 1.0);
        let e: Vec<Pose> = r
            .iter()
            .map(|p| Pose {
                t: p.t,
                p: g * p.p + Vector3::new(5.0, -2.0, 0.3),
                q: g * p.q,
            })
            .collect();
        let s = score(&e, &r, 0.01);
        assert!((s.completeness - 1.0).abs() < 1e-12);
        assert!(s.raw_ate_rmse_m.unwrap() < 1e-9);
        assert!(s.gap_adjusted_ate_rmse_m < 1e-9);
        assert_eq!(s.bucket, "easy");
        assert!((s.mean_rotation_deg_s - 30.0).abs() < 0.5);
    }

    #[test]
    fn gaps_lower_completeness_and_raise_the_gap_adjusted_error() {
        let r = path(20.0, 70.0);
        // Lost from 5 s to 15 s.
        let e: Vec<Pose> = r
            .iter()
            .filter(|p| p.t < 5.0 || p.t > 15.0)
            .copied()
            .collect();
        let s = score(&e, &r, 0.01);
        assert!((s.completeness - 0.5).abs() < 0.02, "{}", s.completeness);
        assert_eq!(s.raw_ate_rmse_m, None, "under 85 % is reported as X");
        assert!(
            s.gap_adjusted_ate_rmse_m > 0.5,
            "{}",
            s.gap_adjusted_ate_rmse_m
        );
        assert_eq!(s.bucket, "hard");
    }
}
