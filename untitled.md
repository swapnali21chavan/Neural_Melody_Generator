Step 1 & Step 2
Progress Log — Setup Phase
Created project folder and conda environment — set up a local conda environment (./env) inside the project directory with Python and core packages (numpy, matplotlib, jupyter).
Installed PyTorch — hit a conda archive-extraction bug (Windows-specific, likely antivirus interference) after clearing the package cache didn't help; switched to installing PyTorch via pip from PyTorch's official CPU wheel index instead, which worked cleanly.
Installed remaining ML/music libraries — music21, pretty_midi, and jupyter for MIDI parsing and notebook work.
Verified the environment — ran a test notebook (01_setup_check.ipynb) importing torch, music21, pretty_midi, and numpy to confirm everything works together.
Installed Git (wasn't on the system yet) and initialized version control — git init in the project folder, added a .gitignore to exclude the environment folder and MIDI data files from version control.
Set up GitHub remote and pushed first commit — created the Neural_Melody_Generator repository on GitHub, connected it via SSH/HTTPS auth, and pushed the initial commit containing the environment setup and verification notebook.