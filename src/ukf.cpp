#include "ukf.h"
#include "Eigen/Dense"
#include <math.h>
#include <iostream>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// +====================================+
// | Initialize Unscented Kalman filter |
// +====================================+
UKF::UKF()
{

  // initially set to false, set to true in first call of ProcessMeasurement
  // car state and covariance matrices will be initialized using the first measurement
  is_initialized_ = false;

  use_laser_ = true; // if false, ignore laser measurements (except during init)
  use_radar_ = true; // if false, ignore radar measurements (except during init)

  // +----------------------------------------------------------------+
  // | Measurement noise values (provided by the sensor manufacturer) |
  // +----------------------------------------------------------------+
  std_laspx_ = 0.15; // LIDAR measurement x-position standard deviation in m
  std_laspy_ = 0.15; // LIDAR measurement y-position standard deviation in m

  std_radr_ = 0.3;    // Radar measurement radius standard deviation in m
  std_radphi_ = 0.03; // Radar measurement angle standard deviation in rad
  std_radrd_ = 0.3;   // Radar measurement radius change standard deviation in m/s

  n_x_ = 5;          // State dimension
  n_aug_ = n_x_ + 2; // Augmented state dimension

  x_ = VectorXd(n_x_);       // initial state vector
  P_ = MatrixXd(n_x_, n_x_); // initial covariance matrix
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
  		0, 0, 0, 1000, 0,
  		0, 0, 0, 0, 1000;
        //0, 0, 0, std_laspx_*std_laspx_, 0,
        //0, 0, 0, 0, std_laspy_*std_laspy_;

  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Process Noise Parameters for the CTRV Model
  std_a_ = 3;          // longitudinal acceleration (a.k.a. linear acceleration) noise in m/s^2 (orig: 30)
  std_yawdd_ = 2*M_PI; // yaw acceleration (a.k.a. angular acceleration) noise in rad/s^2 (orig: 30)

  // time when the state is true, in us
  time_us_ = 0.0;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);

  // Initialize additive measurement covariances
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0, std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
               0, std_laspy_*std_laspy_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage measurement_package)
{

  // Since we don't know where the vehicle is until we receive the first measurement,
  // we will wait until the first measurement arrives before initializing the state vector
  if (!is_initialized_) // if this is the first measurement ...
  {
    // then initialize the state and covariance matrices based on the first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho     = measurement_package.raw_measurements_[0]; // range
      double phi     = measurement_package.raw_measurements_[1]; // bearing
      double rho_dot = measurement_package.raw_measurements_[2]; // velocity in radial direction

      double x = rho * cos(phi);
      double y = rho * sin(phi);

      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      // Set the state vector with the initial position and velocity transforms
      x_ << rho * cos(phi), // x-position
      		rho * sin(phi), // y-position
      		v, 
      		0, 
      		0;
    }
    else // (measurement_package.sensor_type_ == MeasurementPackage::LASER)
    {
      // Set the state vector with the initial location and zero velocity
      x_ << measurement_package.raw_measurements_[0], // x-position
            measurement_package.raw_measurements_[1], // y-position
            0,
            0,
            0;
    }
    is_initialized_ = true;
    return; // filtering starts on the second frame
  }

  // If this is not the first measurement, then run Prediction on deltaTime,
  // followed by an Update to the track state and covariance based on measurements
  
  // Compute the elapsed time between the current and previous measurements
  double deltaTime = (meas_package.timestamp_ - time_us_) / 1000000.0; // deltaTime is in seconds
  time_us_ = meas_package.timestamp_;

  // predict
  Prediction(deltaTime);

  // update
  if (measurement_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    UpdateRadar(measurement_package);

  if (measurement_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    UpdateLidar(measurement_package);
}

void UKF::Prediction(double delta_t)
{
  // Augment the state estimation vector and covariance matrix by adding the nonlinear process noise nu_k
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Generate Sigma Points that sample the augmented state & covariance matrix
  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p, v_p, yaw_p, yawd_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    v_p = v;
    yaw_p = yaw + yawd * delta_t;
    yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawd * delta_t;

    // write predicted sigma points
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covairance
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * Use lidar data to update the belief about the object's position. 
   * Modify the state vector, x_, and covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // combine predicted state and measured state to obtain an updated location

  // extract measurement
  VectorXd z_ = meas_package.raw_measurements_;

  int n_z_ = 2;
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  // predict mean measurement
  VectorXd z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
  }

  // calculate covariance of predicted measurement
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred_;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // additive noise
  S = S + R_lidar_;

  // UKF update
  // Cross correlation matix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred_;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance
  // residual
  VectorXd z_diff = z_ - z_pred_;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  double NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * Use radar data to update the belief about the object's position. 
   * Modify the state vector, x_, and covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Measurement prediction
  // set measurement dimension, radar can measure r, phi, and r_dot

  //extract measurement as VectorXd
  VectorXd z_ = meas_package.raw_measurements_;

  int n_z_ = 3;
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double yawd = Xsig_pred_(4, i);

    double vx = cos(yaw) * v;
    double vy = sin(yaw) * v;

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                           // r
    Zsig(1, i) = atan2(p_y, p_x);                                       // phi
    Zsig(2, i) = (p_x * vx + p_y * vy) / (sqrt(p_x * p_x + p_y * p_y)); // r_dot
  }

  // calculate mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
  }

  // calculate covariance of predicted measurement
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred_;

    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;

  // UKF update
  // Cross correlation matrixc between sigma points in state space
  // and measurement space
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred_;
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance
  // residual
  VectorXd z_diff = z_ - z_pred_;
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //calculate NIS
  double NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}