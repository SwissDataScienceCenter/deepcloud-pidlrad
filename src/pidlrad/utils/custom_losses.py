import torch
from torchmetrics import Metric, MeanAbsoluteError, MeanSquaredError
from .heating_rate import calculate_heating_rates

class EnergyConservationLossV1(Metric):
  def __init__(self, alpha=0.5, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.alpha = alpha
    self.add_state("mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("ec_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

  def hr(self, y):
    return (y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
      (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])

  def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
    mse_loss = torch.sum(torch.square(y_true - y_pred))
    ec_loss = torch.sum(torch.square(self.hr(y_true) - self.hr(y_pred)))
    total_loss = self.alpha * mse_loss + (1 - self.alpha) * ec_loss
    
    self.mse_loss += mse_loss
    self.ec_loss += ec_loss
    self.total_loss += total_loss
    self.total += y_true.shape[0]

  def compute(self):
    return self.total_loss / self.total


class EnergyConservationLossV2(Metric):
  def __init__(self, alpha=0.5, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.alpha = alpha
    self.add_state("mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("ec_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

  def hr(self, y, p):
    dy = ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) -
          (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) 
    dp = (p[:, :-1] - p[:, 1:])
    return dy / (dp + 1e-9)

  def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, pres: torch.Tensor):
    mse_loss = torch.mean(torch.square(y_true - y_pred))
    ec_loss = torch.mean(torch.square(self.hr(y_true, pres) - self.hr(y_pred, pres)))
    total_loss = self.alpha * mse_loss + (1 - self.alpha) * ec_loss
    
    self.mse_loss += mse_loss
    self.ec_loss += ec_loss
    self.total_loss += total_loss
    self.total += y_true.shape[0]

  def compute(self):
    return self.total_loss / self.total


class NormalizedMeanSquaredError(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
  def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
    mse_loss = torch.mean(torch.square((y_true - y_pred)/(y_pred + 1e-9)))
    self.mse_loss += mse_loss
    self.total += y_pred.shape[0]

  def compute(self):
    return self.mse_loss / self.total


class EnergyConservationLossV3(EnergyConservationLossV1):
  def hr(self, y):
    return ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
      (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]]))**2


class EnergyConservationLossV4(EnergyConservationLossV2):
  def hr(self, y, p):
    dy = ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) -
          (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) 
    dp = (p[:, :-1] - p[:, 1:])
    return dy**2 / (dp**2 + 1e-9)


class ColumnMae(Metric):
  def __init__(self, alpha=0.5, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.alpha = alpha
    self.add_state("y_mae", default=torch.zeros(71, 4), dist_reduce_fx="sum")
    self.add_state("hr_mae", default=torch.zeros(70, 2), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

  def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, x3d: torch.Tensor, x2d: torch.Tensor):
    
    hr_true = calculate_heating_rates(y_true, x3d, x2d)
    hr_pred = calculate_heating_rates(y_pred, x3d, x2d)
    
    self.y_mae += torch.sum(torch.abs(y_true - y_pred), dim=0)
    self.hr_mae += torch.sum(torch.abs(hr_true - hr_pred), dim=0)
    self.total += y_true.shape[0]

  def compute(self):
    return self.y_mae / self.total, self.hr_mae / self.total


def get_loss(loss_name):
  loss_classes = {
    'mae': MeanAbsoluteError(),
    'mse': MeanSquaredError(),
    'mean_squared_error': MeanSquaredError(),
    'nmse': NormalizedMeanSquaredError(),
    'normalized_mean_squared_error': NormalizedMeanSquaredError(),
    'eclv1': EnergyConservationLossV1(),
    'eclv2': EnergyConservationLossV2(),
    'eclv3': EnergyConservationLossV3(),
    'eclv4': EnergyConservationLossV4(),
    'column_mae': ColumnMae(),
  }

  loss = loss_classes.get(loss_name)
  if loss is None:
    raise NotImplementedError(f'Loss {loss_name} not implemented!')
  return loss
