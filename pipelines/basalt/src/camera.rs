//! Basalt's Double Sphere camera model, and Basalt's Kannala-Brandt (`kb4`) fisheye model
//! for cameras calibrated that way (Trinet's factory calibration is `kb4`). A camera is one or
//! the other: when [`DoubleSphereCamera::kb4`] is set, every projection below takes the
//! Kannala-Brandt path and `xi`/`alpha` are unused; when it is not, the Double Sphere code runs
//! exactly as before, keeping parity with the pinned upstream.
//!
//! The equations here follow the Double Sphere model used by Basalt at the
//! pinned upstream revision in [`crate::provenance`]. A camera-frame point is
//! projected into EuRoC pixel coordinates; an unprojection returns a unit ray
//! in that same camera frame. Image bounds are not imposed by `project`, so a
//! valid point outside the sensor rectangle can still be used by an optimizer.

use nalgebra::{Matrix2x3, Point2, Point3, Vector2, Vector3};
use thiserror::Error;

const EPS: f64 = 1e-12;

/// Parameters and image dimensions for a Double Sphere camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub xi: f64,
    pub alpha: f64,
    pub width: u32,
    pub height: u32,
    /// Kannala-Brandt coefficients and valid range; `None` for Double Sphere.
    pub kb4: Option<Kb4>,
}

/// Kannala-Brandt `[k1, k2, k3, k4]` and the largest angle off the optical axis the model
/// describes: where `theta_d(theta)` stops increasing (a calibrated polynomial turns back past
/// the lens's field of view, and beyond that point the inverse is not unique).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kb4 {
    pub k: [f64; 4],
    pub theta_max: f64,
}

impl Kb4 {
    pub fn new(k: [f64; 4]) -> Self {
        let slope = |t: f64| DoubleSphereCamera::kb_distort(&k, t).1;
        let pi = std::f64::consts::PI;
        let step = pi / 720.0;
        let mut theta_max = pi;
        let mut a = 0.0;
        while a < pi {
            let b = (a + step).min(pi);
            if slope(b) <= 0.0 {
                // Bisect the first sign change of the slope.
                let (mut lo, mut hi) = (a, b);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if slope(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                theta_max = lo;
                break;
            }
            a = b;
        }
        Self { k, theta_max }
    }
}

impl DoubleSphereCamera {
    /// Constructs and validates a Double Sphere calibration.
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        xi: f64,
        alpha: f64,
        width: u32,
        height: u32,
    ) -> Result<Self, CameraModelError> {
        let values = [fx, fy, cx, cy, xi, alpha];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CameraModelError::NonFiniteParameter);
        }
        if fx <= 0.0 || fy <= 0.0 {
            return Err(CameraModelError::NonPositiveFocalLength { fx, fy });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(CameraModelError::AlphaOutOfRange { alpha });
        }
        if width == 0 || height == 0 {
            return Err(CameraModelError::InvalidResolution { width, height });
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            xi,
            alpha,
            width,
            height,
            kb4: None,
        })
    }

    /// Constructs a Kannala-Brandt (`kb4`) fisheye: `theta_d = theta (1 + k1 theta^2 + k2
    /// theta^4 + k3 theta^6 + k4 theta^8)`, the model OpenCV calls fisheye.
    #[allow(clippy::too_many_arguments)]
    pub fn new_kannala_brandt4(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k: [f64; 4],
        width: u32,
        height: u32,
    ) -> Result<Self, CameraModelError> {
        if [fx, fy, cx, cy]
            .iter()
            .chain(k.iter())
            .any(|value| !value.is_finite())
        {
            return Err(CameraModelError::NonFiniteParameter);
        }
        if fx <= 0.0 || fy <= 0.0 {
            return Err(CameraModelError::NonPositiveFocalLength { fx, fy });
        }
        if width == 0 || height == 0 {
            return Err(CameraModelError::InvalidResolution { width, height });
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            xi: 0.0,
            alpha: 0.0,
            width,
            height,
            kb4: Some(Kb4::new(k)),
        })
    }

    fn kb_distort(k: &[f64; 4], theta: f64) -> (f64, f64) {
        let t2 = theta * theta;
        let d = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
        let dd = 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
        (d, dd)
    }

    fn kb_project(&self, m: &Kb4, point: &Point3<f64>) -> Option<Point2<f64>> {
        self.kb_project_with_jacobian(m, &point.coords)
            .map(|(p, _)| Point2::new(p.x, p.y))
    }

    /// Kannala-Brandt projection and its 2×3 Jacobian with respect to the camera-frame point.
    fn kb_project_with_jacobian(
        &self,
        m: &Kb4,
        p: &Vector3<f64>,
    ) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
        let k = &m.k;
        if !p.iter().all(|value| value.is_finite()) {
            return None;
        }
        let (x, y, z) = (p.x, p.y, p.z);
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let rho2 = r2 + z * z;
        if rho2 <= EPS * EPS {
            return None;
        }
        let mut j = Matrix2x3::zeros();
        if r <= 1e-9 {
            // On the optical axis the model is pinhole to first order.
            if z <= EPS {
                return None;
            }
            j[(0, 0)] = self.fx / z;
            j[(1, 1)] = self.fy / z;
            j[(0, 2)] = -self.fx * x / (z * z);
            j[(1, 2)] = -self.fy * y / (z * z);
            return Some((
                Vector2::new(self.fx * x / z + self.cx, self.fy * y / z + self.cy),
                j,
            ));
        }
        let theta = r.atan2(z);
        if theta > m.theta_max {
            return None;
        }
        let (d, dd) = Self::kb_distort(k, theta);
        let s = d / r;
        let u = Vector2::new(self.fx * x * s + self.cx, self.fy * y * s + self.cy);
        // ds/dq = (d'(θ) dθ/dq · r − d · dr/dq) / r²
        let dtheta = Vector3::new(x * z / (r * rho2), y * z / (r * rho2), -r / rho2);
        let dr = Vector3::new(x / r, y / r, 0.0);
        let ds = (dtheta * (dd * r) - dr * d) / r2;
        for c in 0..3 {
            j[(0, c)] = self.fx * (x * ds[c] + if c == 0 { s } else { 0.0 });
            j[(1, c)] = self.fy * (y * ds[c] + if c == 1 { s } else { 0.0 });
        }
        (u.iter().chain(j.iter()).all(|v| v.is_finite())).then_some((u, j))
    }

    fn kb_unproject_raw(&self, m: &Kb4, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let mx = (pixel.x - self.cx) / self.fx;
        let my = (pixel.y - self.cy) / self.fy;
        let theta_d = (mx * mx + my * my).sqrt();
        if theta_d <= 1e-12 {
            return Some(Vector3::new(0.0, 0.0, 1.0));
        }
        // theta_d(theta) increases on [0, theta_max]: outside its range the pixel has no ray.
        let (d_max, _) = Self::kb_distort(&m.k, m.theta_max);
        if theta_d > d_max {
            return None;
        }
        // Newton kept inside a shrinking bracket (bisection when a step would leave it).
        let (mut lo, mut hi) = (0.0, m.theta_max);
        let mut theta = theta_d.min(m.theta_max);
        for _ in 0..60 {
            let (d, dd) = Self::kb_distort(&m.k, theta);
            let f = d - theta_d;
            if f.abs() < 1e-13 {
                break;
            }
            if f > 0.0 {
                hi = theta;
            } else {
                lo = theta;
            }
            let newton = theta - f / dd;
            theta = if dd > 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
        }
        let sin_over = theta.sin() / theta_d;
        Some(Vector3::new(mx * sin_over, my * sin_over, theta.cos()))
    }

    /// Projection and its Jacobian with respect to the camera-frame point, for Kannala-Brandt
    /// cameras; `None` for Double Sphere ones (their Jacobians live with the optimizer).
    pub fn kb4_project_with_jacobian(
        &self,
        point: &Vector3<f64>,
    ) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
        let m = self.kb4?;
        self.kb_project_with_jacobian(&m, point)
    }

    /// Projects a camera-frame point into pixel coordinates.
    ///
    /// The point must have a finite, non-zero norm and a positive Double
    /// Sphere denominator. The result is not clipped to image bounds.
    pub fn project(&self, point: &Point3<f64>) -> Option<Point2<f64>> {
        if let Some(m) = self.kb4 {
            return self.kb_project(&m, point);
        }
        if !point.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let d1 = point.coords.norm();
        if d1 <= EPS {
            return None;
        }

        let zeta = self.xi * d1 + point.z;
        let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
        let denominator = self.alpha * d2 + (1.0 - self.alpha) * zeta;
        if !denominator.is_finite() || denominator <= EPS {
            return None;
        }

        let pixel = Point2::new(
            self.fx * point.x / denominator + self.cx,
            self.fy * point.y / denominator + self.cy,
        );
        pixel
            .coords
            .iter()
            .all(|value| value.is_finite())
            .then_some(pixel)
    }

    /// Float32-owned projection used by the upstream compatibility core.
    /// Parameters are cast at the API boundary and every intermediate below
    /// is an f32, matching `GenericCamera<float>` rather than merely casting
    /// the final pixel produced by the f64 implementation.
    pub fn project_f32(&self, point: &Point3<f32>) -> Option<Point2<f32>> {
        if let Some(m) = self.kb4 {
            let p = Point3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z));
            return self
                .kb_project(&m, &p)
                .map(|q| Point2::new(q.x as f32, q.y as f32));
        }
        if !point.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let fx = self.fx as f32;
        let fy = self.fy as f32;
        let cx = self.cx as f32;
        let cy = self.cy as f32;
        let xi = self.xi as f32;
        let alpha = self.alpha as f32;
        let d1 = point.coords.norm();
        if d1 <= f32::EPSILON {
            return None;
        }
        let zeta = xi * d1 + point.z;
        let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
        let denominator = alpha * d2 + (1.0_f32 - alpha) * zeta;
        if !denominator.is_finite() || denominator <= f32::EPSILON {
            return None;
        }
        let pixel = Point2::new(
            fx * point.x / denominator + cx,
            fy * point.y / denominator + cy,
        );
        pixel
            .coords
            .iter()
            .all(|value| value.is_finite())
            .then_some(pixel)
    }

    /// Unprojects a pixel into a unit camera-frame ray.
    pub fn unproject(&self, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        let ray = self.unproject_raw(pixel)?;
        let norm = ray.norm();
        if !norm.is_finite() || norm <= EPS {
            return None;
        }
        Some(ray / norm)
    }

    /// Unprojects a pixel into Basalt's raw camera-frame vector.  Upstream's
    /// DLT triangulator consumes this non-unit vector directly; the public
    /// [`Self::unproject`] method continues to expose the normalized ray used
    /// by the factor/geometry APIs.
    pub fn unproject_raw(&self, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        if let Some(m) = self.kb4 {
            return self.kb_unproject_raw(&m, pixel);
        }
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return None;
        }

        let mx = (pixel.x - self.cx) / self.fx;
        let my = (pixel.y - self.cy) / self.fy;
        let r_sq = mx * mx + my * my;
        let sqrt_argument = 1.0 - (2.0 * self.alpha - 1.0) * r_sq;
        if !sqrt_argument.is_finite() || sqrt_argument < 0.0 {
            return None;
        }

        let mz_denominator = self.alpha * sqrt_argument.sqrt() + (1.0 - self.alpha);
        if mz_denominator.abs() <= EPS {
            return None;
        }
        let mz = (1.0 - self.alpha * self.alpha * r_sq) / mz_denominator;

        let k_radicand = mz * mz + (1.0 - self.xi * self.xi) * r_sq;
        if !k_radicand.is_finite() || k_radicand < 0.0 {
            return None;
        }
        let k_denominator = mz * mz + r_sq;
        if k_denominator <= EPS {
            return None;
        }
        let k = (mz * self.xi + k_radicand.sqrt()) / k_denominator;
        // The inverse first recovers the sphere point `k * [mx, my, mz]`;
        // subtracting xi on z returns the original camera-frame ray.
        let ray = Vector3::new(k * mx, k * my, k * mz - self.xi);
        if !ray.iter().all(|value| value.is_finite()) || ray.norm() <= EPS {
            return None;
        }
        Some(ray)
    }

    /// Float32-owned raw double-sphere unprojection.  This is deliberately a
    /// separate path so observation bearings and triangulation retain the
    /// upstream storage/rounding boundary all the way through the formulas.
    pub fn unproject_raw_f32(&self, pixel: &Point2<f32>) -> Option<Vector3<f32>> {
        if let Some(m) = self.kb4 {
            let p = Point2::new(f64::from(pixel.x), f64::from(pixel.y));
            return self
                .kb_unproject_raw(&m, &p)
                .map(|v| Vector3::new(v.x as f32, v.y as f32, v.z as f32));
        }
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let fx = self.fx as f32;
        let fy = self.fy as f32;
        let cx = self.cx as f32;
        let cy = self.cy as f32;
        let xi = self.xi as f32;
        let alpha = self.alpha as f32;
        let mx = (pixel.x - cx) / fx;
        let my = (pixel.y - cy) / fy;
        // Eigen's scalar packet path contracts the final square/addition
        // (`mx * mx + my * my`) into one FMA. Keep this reduction explicit;
        // it is the first expression-order boundary that can differ from a
        // generic f32 translation.
        let r_sq = mx.mul_add(mx, my * my);
        // Keep the scalar temporaries and expression order of
        // DoubleSphereCamera<float>::unproject. The pinned native build
        // contracts the scalar reductions with FMA; algebraically equivalent
        // rewrites can differ by one or more f32 ulps here.
        let xi2_2 = alpha * alpha;
        let xi1_2 = xi * xi;
        let sqrt2 = (-(2.0_f32 * alpha - 1.0_f32)).mul_add(r_sq, 1.0_f32).sqrt();
        if !sqrt2.is_finite() {
            return None;
        }
        // The header writes `alpha * sqrt2 + Scalar(1) - alpha`; retain its
        // left-associative additions rather than evaluating `(1 - alpha)`
        // first, while preserving the native multiply-add contraction.
        let norm2 = alpha.mul_add(sqrt2, 1.0_f32) - alpha;
        if !norm2.is_finite() || norm2.abs() <= f32::EPSILON {
            return None;
        }
        // GCC's native upstream build contracts the numerator subtraction
        // into `1 - alpha^2 * r_sq` and the subsequent norm/sqrt reductions
        // into the same scalar FMA order.
        let mz = (-xi2_2).mul_add(r_sq, 1.0_f32) / norm2;
        let norm1 = mz.mul_add(mz, r_sq);
        if !norm1.is_finite() || norm1 <= f32::EPSILON {
            return None;
        }
        let sqrt1 = mz.mul_add(mz, (1.0_f32 - xi1_2) * r_sq).sqrt();
        if !sqrt1.is_finite() {
            return None;
        }
        let k = mz.mul_add(xi, sqrt1) / norm1;
        let ray = Vector3::new(k * mx, k * my, k.mul_add(mz, -xi));
        if !ray.iter().all(|value| value.is_finite()) || ray.norm() <= f32::EPSILON {
            return None;
        }
        Some(ray)
    }

    pub fn unproject_f32(&self, pixel: &Point2<f32>) -> Option<Vector3<f32>> {
        let ray = self.unproject_raw_f32(pixel)?;
        let norm = ray.norm();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return None;
        }
        Some(ray / norm)
    }

    /// Returns whether a pixel lies inside the calibrated image rectangle.
    pub fn contains_pixel(&self, pixel: &Point2<f64>) -> bool {
        pixel.x >= 0.0
            && pixel.x < self.width as f64
            && pixel.y >= 0.0
            && pixel.y < self.height as f64
    }
}

/// Validation failures for Double Sphere parameters.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum CameraModelError {
    #[error("Double Sphere parameters contain a non-finite value")]
    NonFiniteParameter,
    #[error("focal lengths must be positive, got fx={fx}, fy={fy}")]
    NonPositiveFocalLength { fx: f64, fy: f64 },
    #[error("Double Sphere alpha must lie in [0, 1], got {alpha}")]
    AlphaOutOfRange { alpha: f64 },
    #[error("image resolution must be non-zero, got {width}x{height}")]
    InvalidResolution { width: u32, height: u32 },
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use super::*;

    fn camera() -> DoubleSphereCamera {
        DoubleSphereCamera::new(
            458.654, 457.296, 367.215, 248.375, 0.662073, 0.779792, 752, 480,
        )
        .unwrap()
    }

    #[test]
    fn projection_matches_double_sphere_golden_value() {
        let pixel = camera().project(&Point3::new(0.1, -0.05, 1.0)).unwrap();
        assert!((pixel.x - 394.693_793_482_414_3).abs() < 1e-9, "{pixel:?}");
        assert!((pixel.y - 234.676_283_381_008_18).abs() < 1e-9, "{pixel:?}");
    }

    #[test]
    fn projection_and_unprojection_preserve_ray_direction() {
        let point = Point3::new(0.1, -0.05, 1.0);
        let pixel = camera().project(&point).unwrap();
        let ray = camera().unproject(&pixel).unwrap();
        let expected = point.coords.normalize();
        assert!(
            (ray - expected).norm() < 1e-10,
            "ray={ray:?}, expected={expected:?}"
        );
    }

    #[test]
    fn raw_f32_unprojection_preserves_pinned_expression_bits() {
        let camera = DoubleSphereCamera::new(
            349.7560023050409,
            348.72454229977037,
            365.89440762590149,
            249.32995565708704,
            -0.2409573942178872,
            0.566996899163044,
            752,
            480,
        )
        .unwrap();
        let ray = camera
            .unproject_raw_f32(&Point2::new(97.0_f32, 258.0_f32))
            .unwrap();
        assert_eq!(ray.x.to_bits(), 0xbf0c_f651);
        assert_eq!(ray.y.to_bits(), 0x3c91_df79);
        assert_eq!(ray.z.to_bits(), 0x3f55_a58a);
    }

    #[test]
    fn m7_track2_raw_f32_unprojection_matches_pinned_native_bits() {
        let camera0 = DoubleSphereCamera::new(
            349.7560023050409,
            348.72454229977037,
            365.89440762590149,
            249.32995565708704,
            -0.2409573942178872,
            0.566996899163044,
            752,
            480,
        )
        .unwrap();
        let camera1 = DoubleSphereCamera::new(
            361.6713883800533,
            360.5856493689301,
            379.40818394080869,
            255.9772968522045,
            -0.21300835384809328,
            0.5767008625037023,
            752,
            480,
        )
        .unwrap();
        let ray0 = camera0
            .unproject_raw_f32(&Point2::new(46.0_f32, 118.0_f32))
            .unwrap();
        let ray1 = camera1
            .unproject_raw_f32(&Point2::new(52.372352600097656, 132.4171142578125))
            .unwrap();
        assert_eq!(
            [ray0.x.to_bits(), ray0.y.to_bits(), ray0.z.to_bits()],
            [0xbf21_4708, 0xbe84_d05b, 0x3f3b_6451]
        );
        assert_eq!(
            [ray1.x.to_bits(), ray1.y.to_bits(), ray1.z.to_bits()],
            [0xbf24_c3d6, 0xbe79_c13a, 0x3f39_b6ef]
        );
    }

    #[test]
    fn invalid_parameters_and_degenerate_points_are_rejected() {
        assert!(matches!(
            DoubleSphereCamera::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1, 1),
            Err(CameraModelError::NonPositiveFocalLength { .. })
        ));
        assert!(camera().project(&Point3::origin()).is_none());
        assert!(camera().unproject(&Point2::new(f64::NAN, 0.0)).is_none());
    }

    /// Trinet Pro Stereo's left eye (factory calibration, 1920×1080).
    fn trinet() -> DoubleSphereCamera {
        DoubleSphereCamera::new_kannala_brandt4(
            589.225_280_761_718_8,
            591.105_712_890_625,
            862.521_423_339_843_8,
            566.608_459_472_656_2,
            [0.154_377_371, 0.032_201_048, -0.050_931_457, 0.007_933_425],
            1920,
            1080,
        )
        .unwrap()
    }

    #[test]
    fn kb4_round_trips_out_to_the_fisheye_edge() {
        let cam = trinet();
        for deg in [0.0f64, 5.0, 30.0, 60.0, 80.0, 95.0] {
            for az in [0.0f64, 1.0, 2.5, 4.0] {
                let t = deg.to_radians();
                let ray = Vector3::new(t.sin() * az.cos(), t.sin() * az.sin(), t.cos());
                let px = cam.project(&Point3::from(ray * 3.0)).unwrap();
                let back = cam.unproject(&px).unwrap();
                assert!(
                    (back - ray).norm() < 1e-9,
                    "{deg}° az {az}: {back:?} vs {ray:?}"
                );
            }
        }
    }

    #[test]
    fn kb4_jacobian_matches_finite_differences() {
        let cam = trinet();
        for p in [
            Vector3::new(0.3, -0.2, 1.5),
            Vector3::new(-1.2, 0.4, 0.6),
            Vector3::new(0.9, 1.1, -0.1),
            Vector3::new(1e-12, 0.0, 2.0),
        ] {
            let (u, j) = cam.kb4_project_with_jacobian(&p).unwrap();
            for c in 0..3 {
                let h = 1e-6;
                let mut q = p;
                q[c] += h;
                let (uq, _) = cam.kb4_project_with_jacobian(&q).unwrap();
                let numeric = (uq - u) / h;
                assert!(
                    (numeric - j.column(c)).norm() < 1e-3 * (1.0 + numeric.norm()),
                    "{p:?} column {c}: {numeric:?} vs {:?}",
                    j.column(c)
                );
            }
        }
    }

    #[test]
    fn kb4_refuses_rays_past_the_calibrated_field() {
        let cam = trinet();
        let m = cam.kb4.unwrap();
        assert!(
            (m.theta_max.to_degrees() - 99.6).abs() < 0.5,
            "{}",
            m.theta_max.to_degrees()
        );
        let t = 110f64.to_radians();
        assert!(cam.project(&Point3::new(t.sin(), 0.0, t.cos())).is_none());
        // The right-hand image corners lie beyond what the factory polynomial describes.
        assert!(cam.unproject(&Point2::new(1919.0, 0.0)).is_none());
        assert!(cam.unproject(&Point2::new(0.0, 566.0)).is_some());
    }

    #[test]
    fn a_ds_camera_has_no_kb4_path() {
        assert!(camera().kb4.is_none());
        assert!(camera()
            .kb4_project_with_jacobian(&Vector3::new(0.1, 0.1, 1.0))
            .is_none());
    }
}
