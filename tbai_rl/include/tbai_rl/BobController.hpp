#ifndef TBAI_RL_INCLUDE_TBAI_RL_BOBCONTROLLER_HPP_
#define TBAI_RL_INCLUDE_TBAI_RL_BOBCONTROLLER_HPP_

// clang-format off
#include <pinocchio/fwd.hpp>
// clang-format on

#include <memory>
#include <string>
#include <vector>

#include "tbai_rl/CentralPatternGenerator.hpp"
#include "tbai_rl/InverseKinematics.hpp"
#include <Eigen/Dense>
#include <tbai_core/control/Controller.hpp>
#include <tbai_core/control/StateSubscriber.hpp>
#include <tbai_gridmap/GridmapInterface.hpp>
#include <tbai_reference/ReferenceVelocityGenerator.hpp>
#include <torch/script.h>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <functional>

#include <tbai_rl/State.hpp>
#include <tbai_rl/Visualizers.hpp>

namespace tbai {
namespace rl {

using namespace tbai;
using namespace torch::indexing;
using torch::jit::script::Module;

class BobController : public tbai::core::Controller {
   public:
    BobController(const std::shared_ptr<tbai::core::StateSubscriber> &stateSubscriberPtr);

    tbai_msgs::JointCommandArray getCommandMessage(scalar_t currentTime, scalar_t dt) override;

    void visualize() override;

    void changeController(const std::string &controllerType, scalar_t currentTime) override;

    bool isSupported(const std::string &controllerType) override;

    void stopController() override {}

    scalar_t getRate() const override { return 50.0; }

   private:
    std::shared_ptr<tbai::core::StateSubscriber> stateSubscriberPtr_;

    scalar_t kp_;
    scalar_t kd_;

    std::unique_ptr<InverseKinematics> ik_;
    std::unique_ptr<CentralPatternGenerator> cpg_;
    std::unique_ptr<tbai::reference::ReferenceVelocityGenerator> refVelGen_;
    std::unique_ptr<tbai::gridmap::GridmapInterface> gridmap_;

    /* Copied section */

    scalar_t LIN_VEL_SCALE = 2.0;
    scalar_t ANG_VEL_SCALE = 0.25;
    scalar_t GRAVITY_SCALE = 1.0;
    scalar_t COMMAND_SCALE = 1.0;
    scalar_t JOINT_POS_SCALE = 1.0;
    scalar_t JOINT_VEL_SCALE = 0.05;
    scalar_t ACTION_SCALE = 0.5;
    scalar_t HEIGHT_MEASUREMENTS_SCALE = 1.0;

    int POSITION_HISTORY_SIZE = 3;
    int VELOCITY_HISTORY_SIZE = 2;
    int COMMAND_HISTORY_SIZE = 2;

    int POSITION_SIZE = 12;
    int VELOCITY_SIZE = 12;
    int COMMAND_SIZE = 12;

    Module model_;

    constexpr size_t getNNInputSize() { return 3 + 3 + 3 + 3 + 12 + 12 + 3 * 12 + 2 * 12 + 2 * 12 + 8 + 4 * 52; }

    at::Tensor getNNInput(const State &state, scalar_t currentTime, scalar_t dt);

    void setupPinocchioModel();
    tbai_msgs::JointCommandArray getCommandMessage(const vector_t &jointAngles);
    pinocchio::Model pinocchioModel_;
    pinocchio::Data pinocchioData_;
    State getBobnetState();

    void fillCommand(at::Tensor &input, scalar_t currentTime, scalar_t dt);
    void fillGravity(at::Tensor &input, const State &state);
    void fillBaseLinearVelocity(at::Tensor &input, const State &state);
    void fillBaseAngularVelocity(at::Tensor &input, const State &state);
    void fillJointResiduals(at::Tensor &input, const State &state);
    void fillJointVelocities(at::Tensor &input, const State &state);
    void fillHistory(at::Tensor &input);
    void fillCpg(at::Tensor &input);
    void fillHeights(at::Tensor &input, const State &state);

    void fillHistoryResiduals(at::Tensor &input);
    void fillHistoryVelocities(at::Tensor &input);
    void fillHistoryActions(at::Tensor &input);

    void updateHistory(const at::Tensor &input, const at::Tensor &action, const State &state);
    void resetHistory();

    const Slice commandSlice_ = Slice(0, 3);
    const Slice gravitySlice_ = Slice(3, 6);
    const Slice baseLinearVelocitySlice_ = Slice(6, 9);
    const Slice baseAngularVelocitySlice_ = Slice(9, 12);
    const Slice jointResidualsSlice_ = Slice(12, 24);
    const Slice jointVelocitiesSlice_ = Slice(24, 36);
    const Slice samplesGTSlice_ = Slice(136, None);
    const Slice samplesReconstructedSlice_ = Slice(0, 4 * 52);

    void generateSamplingPositions();

    std::vector<at::Tensor> historyResiduals_;
    int historyResidualsIndex_ = 0;
    std::vector<at::Tensor> historyVelocities_;
    int historyVelocitiesIndex_ = 0;
    std::vector<at::Tensor> historyActions_;
    int historyActionsIndex_ = 0;

    // at::Tensor nnInput_;
    at::Tensor hidden_;

    vector_t jointAngles2_;

    vector_t standJointAngles_;

    matrix_t samplingPositions_;

    matrix_t sampled_;

    std::vector<std::string> jointNames_;

    StateVisualizer stateVisualizer_;
    HeightsReconstructedVisualizer heightsReconstructedVisualizer_;

    bool blind_;
};

}  // namespace rl
}  // namespace tbai

#endif  // TBAI_RL_INCLUDE_TBAI_RL_BOBCONTROLLER_HPP_
