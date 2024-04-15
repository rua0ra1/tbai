#include "tbai_core/CentralController.hpp"

namespace tbai {
namespace core {

/*********************************************************************************************************************/
/*********************************************************************************************************************/
/*********************************************************************************************************************/
void CentralController::addController(std::unique_ptr<Controller> controller, bool makeActive) {
    controllers_.push_back(std::move(controller));
    if (makeActive || activeController_ == nullptr) {
        activeController_ = controllers_.back().get();
    }
}

}  // namespace core
}  // namespace tbai