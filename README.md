# MT Exercise 4: Byte Pair Encoding, Beam Search

This repository is a starting point for the 4th and final exercise. As before, fork this repo to your own account and then clone it into your preferred directory.

---

## Requirements

- Python 3.10 must be installed. The command `python3` (or `python` on Windows) should be available from your terminal or command prompt.
- `virtualenv` must be installed. Install it with:

  ```bash
  pip install virtualenv

macOS/Linux users: No special setup needed; shell scripts should run normally.

Windows users: Either use Windows Subsystem for Linux (WSL) or a Unix-compatible shell like Git Bash.
If you're using PowerShell or Command Prompt, manual setup is required.

Setup Instructions
For macOS / Linux / WSL / Git Bash users
1. Clone your fork of the repository:

git clone https://github.com/[your-username]/mt-exercise-4
cd mt-exercise-4

2. Create a virtual environment:

./scripts/make_virtualenv.sh

    🔹 Important: After running the script, activate the environment using the source command shown at the end. Usually:

source mt_env/bin/activate

3. Install required dependencies

Follow the instructions provided in the exercise PDF.
4. Download data:

./download_iwslt_2017_data.sh

5. Modify the codebase

Make the changes described in the exercise PDF.
6. Train the model:

./scripts/train.sh

    The training process can be interrupted at any time. The best checkpoint will always be saved automatically.

7. Evaluate the model:

./scripts/evaluate.sh

For Windows (Command Prompt / PowerShell users)
1. Manually create and activate a virtual environment:

python -m venv mt_env
mt_env\Scripts\activate

    ⚠ Note: The make_virtualenv.sh script will not work in native Windows shells.

2. Manually download the dataset

Open the download_iwslt_2017_data.sh file in a text editor and run the commands one-by-one in your shell.
Alternatively, use Git Bash or WSL to run it directly.
3. Modify, train, and evaluate

Once setup is complete, use the instructions in the exercise PDF to run training and evaluation (either by adapting the .sh scripts manually, or by using Git Bash/WSL).
Notes for Windows Users

    Using Git Bash or WSL is highly recommended for compatibility.

    If using native PowerShell or Command Prompt:

        Manual recreation of shell script steps will be necessary.

        Always activate your virtual environment before running any training or evaluation steps.

