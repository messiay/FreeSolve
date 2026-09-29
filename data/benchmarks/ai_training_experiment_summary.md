# Side-by-Side AI Training Experiment Summary

Demonstrates the impact of integrating `SolvDockPhysicsLoss` into PyTorch training.

- **Model A (Pure MSE)**: PoseBusters Pass = 5/5 (100.0%), Final Clashes = 1
- **Model B (MSE + SolvDock Physics)**: PoseBusters Pass = 5/5 (100.0%), Final Clashes = 0
- **Clash Reduction**: 1 clashes down to 0 clashes

## Epoch History

| Epoch | Model A MSE | Model A Clashes | Model B MSE | Model B Clashes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.4826 | 5 | 0.5077 | 5 |
| 6 | 0.1129 | 2 | 0.1916 | 0 |
| 11 | 0.0871 | 1 | 0.1828 | 0 |
| 16 | 0.0662 | 1 | 0.1967 | 0 |
| 21 | 0.0396 | 1 | 0.1970 | 0 |
| 26 | 0.0214 | 1 | 0.2008 | 0 |
| 30 | 0.0201 | 1 | 0.2049 | 0 |
