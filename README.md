# Hei Shing Helson Go's C++ math library

This is a repository of my personal implementations of various mathematical
functions as a header-only C++ library.

This library currently contains 4 files, shown below.

```txt
├── include
│   └── hs293go
│       ├── frame_conversions.hpp
│       ├── geo.hpp
│       ├── math.hpp
│       └── rotation.hpp
```

- `frame_conversions.hpp`: Contains functions for converting between NED/ENU
  coordinate frames.
  - Replacement/reimplementation of Mavros's
    [`frame_tf.h`](https://docs.ros.org/en/kinetic/api/mavros/html/frame__tf_8h.html`])
- `geo.hpp`: Contains functions for geodetic calculations
  - Independent porting of a few functions from Matlab's Mapping Toolbox
    - `lla2ecef`: Convert from LLA to ECEF coordinates
    - `enu2ecef`: Convert from ENU to ECEF coordinates
    - `lla2enu`: Convert from LLA to ENU coordinates
- `math.hpp`: Contains a few _bread and butter_ mathematical functions I found
  useful in Matlab or numpy
  - `IsClose`: Check if two floating point numbers are close to each other (from
    python's `math.isclose`)
  - `wrapToPi`: Wrap an angle to the range `[-π, π]` (Matlab builtin, edge
    handling semantics differ)
  - `wrapTo2Pi`: Wrap an angle to the range `[0, 2π]` (Matlab builtin, edge
    handling semantics differ)
  - `wrapTo180`: Wrap an angle to the range `[-180°, 180°]` (Matlab builtin,
    edge handling semantics differ)
  - `wrapTo360`: Wrap an angle to the range `[0°, 360°]` (Matlab builtin, edge
    handling semantics differ)
  - `rad2deg`: Convert radians to degrees (Matlab builtin)
  - `deg2rad`: Convert degrees to radians (Matlab builtin)
- `rotation.hpp`: Contains functions for converting between rotation
  representations
  - Many of these functions are Eigen reimplementations of Ceres's
    [`rotation.h`](https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h)
  - `hat`: Convert a 3D vector to a skew-symmetric matrix
  - `vee`: Convert a skew-symmetric matrix to a 3D vector
  - `QuaternionToAngleAxis`: Convert a quaternion to an angle-axis
    representation
  - `AngleAxisToQuaternion`: Convert an angle-axis representation to a
    quaternion
  - `AngleAxisToRotationMatrix`: Convert an angle-axis representation to a
    rotation matrix
  - `RotationMatrixToAngleAxis`: Convert a rotation matrix to an angle-axis
    representation
  - `EulerAnglesToQuaternion`: Convert aerospace passive world-to-body-ZYX euler
    angles to a passive body-to-world quaternion
  - `QuaternionToEulerAngles`: Convert a passive body-to-world quaternion to
    aerospace passive world-to-body-ZYX euler angles
  - `Negated`: Negate a quaternion by flipping the sign of all components. The
    result represents the same rotation as the input quaternion.
  - `MatchSign`: Match the sign/hemisphere of a quaternion (first) to a
    reference quaternion (second). Useful in trajectory optimization problems to
    ensure continuous quaternion representations.
  - `CanonicalizePositiveW`: Canonicalize a quaternion to have a positive scalar
    component. The result represents the same rotation as the input quaternion.
