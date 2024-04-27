#pragma once

#include <memory>

#include "ocs2_legged_robot_ros/visualization/LeggedRobotVisualizer.h"
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <tbai_core/control/Controller.hpp>
#include <tbai_core/control/StateSubscriber.hpp>
#include <tbai_reference/ReferenceVelocityGenerator.hpp>
#include <torch/script.h>

namespace tbai {
namespace dtc {

class DtcController final : public tbai::core::Controller {
   public:
    DtcController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriber);

    tbai_msgs::JointCommandArray getCommandMessage(scalar_t currentTime, scalar_t dt) override;

    void visualize() override;

    void changeController(const std::string &controllerType, scalar_t currentTime) override;

    bool isSupported(const std::string &controllerType) override;

    void stopController() override {}

    scalar_t getRate() const override { return 50.0; }

   private:
    // Torchscript model
    torch::jit::script::Module dtcModel_;

    ocs2::SystemObservation generateSystemObservation() const;

    void resetMpc();
    void setObservation();

    std::shared_ptr<tbai::core::StateSubscriber> stateSubscriberPtr_;

    const scalar_t LIN_VEL_SCALE = 2.0;
    const scalar_t ANG_VEL_SCALE = 0.25;
    const scalar_t DOF_POS_SCALE = 1.0;
    const scalar_t DOF_VEL_SCALE = 0.05;

    const size_t MODEL_INPUT_SIZE = 107;

    const scalar_t ACTION_SCALE = 0.5;

    void setupPinocchioModel();

    scalar_t initTime_;

    ocs2::MRT_ROS_Interface mrt_;
    bool mrt_initialized_ = false;

    pinocchio::Model model_;
    pinocchio::Data data_;

    scalar_t mpcRate_ = 50.5;
    scalar_t timeSinceLastMpcUpdate_ = 1e5;

    std::unique_ptr<ocs2::legged_robot::LeggedRobotInterface> leggedInterface_;
    std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterface_;
    std::unique_ptr<ocs2::CentroidalModelPinocchioMapping> centroidalModelMapping_;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> endEffectorKinematics_;
    std::unique_ptr<ocs2::legged_robot::LeggedRobotVisualizer> visualizer_;

    std::unique_ptr<tbai::reference::ReferenceVelocityGenerator> refVelGen_;
    void publishReference(scalar_t currentTime, scalar_t dt);
    ros::Publisher refPub_;

    vector_t defaultDofPositions_;

    vector_t obsCommand_;
    vector_t obsLinearVelocity_;
    vector_t obsAngularVelocity_;
    vector_t obsProjectedGravity_;
    vector_t obsJointPositions_;
    vector_t obsJointVelocities_;
    vector_t obsPastAction_;
    vector_t obsDesiredContacts_;
    vector_t obsTimeLeftInPhase_;
    vector_t obsDesiredFootholds_;
    vector_t obsDesiredJointAngles_;
    vector_t obsCurrentDesiredJointAngles_;
    vector_t obsDesiredBasePosition_;
    vector_t obsDesiredBaseOrientation_;
    vector_t obsDesiredBaseLinearVelocity_;
    vector_t obsDesiredBaseAngularVelocity_;
    vector_t obsDesiredBaseLinearAcceleration_;
    vector_t obsDesiredBaseAngularAcceleration_;

    vector_t kk_;

    torch::Tensor action_;

    void initializeObservations();

    // Helper functions for observation generation
    void computeObservation(const ocs2::PrimalSolution &policy, scalar_t time);
    void computeCommandObservation(scalar_t time);
    void computeDesiredContacts(scalar_t time);
    void computeTimeLeftInPhases(scalar_t time);
    void computeDesiredFootholds(scalar_t time);
    void computeCurrentDesiredJointAngles(scalar_t time);
    void computeBaseObservation(scalar_t time);

    void fillObsTensor(torch::Tensor &tensor);
};

}  // namespace dtc
}  // namespace tbai