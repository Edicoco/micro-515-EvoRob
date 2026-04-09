import numpy as np

x = np.load('results/20260324_085356_nsga_ckpts/499/x.npy')          # Population of solutions
# Best Trade-off Individual: 96 with fitness [1255.28576294 1337.85156667]
best_trade_off_idx = 96
best_tradeoff_controller = x[best_trade_off_idx]
# Best Flat Terrain Individual: 123 with fitness [1366.09263751 1020.81232877]
best_individual_flat_idx = 123
best_flat_controller = x[best_individual_flat_idx]
# Best Ice Terrain Individual: 5 with fitness [ 653.14737024 1450.60349035]
best_individual_ice_idx = 5
best_ice_controller = x[best_individual_ice_idx]

#save the best controllers to separate files
np.save('results/best_tradeoff_controller.npy', best_tradeoff_controller)
np.save('results/best_flat_controller.npy', best_flat_controller)
np.save('results/best_ice_controller.npy', best_ice_controller)

best_tradeoff_controller = np.load('results/best_tradeoff_controller.npy')
best_flat_controller = np.load('results/best_flat_controller.npy')
best_ice_controller = np.load('results/best_ice_controller.npy')

print (x[best_trade_off_idx] == best_tradeoff_controller)  # Should print True
print (x[best_individual_flat_idx] == best_flat_controller)  # Should print True
print (x[best_individual_ice_idx] == best_ice_controller)