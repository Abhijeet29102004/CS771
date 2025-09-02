import subprocess

# ANSI escape sequences for colors
GREEN = "\033[92m"  # Green text
RED = "\033[91m"    # Red text
RESET = "\033[0m"   # Reset to default color

# Function to run a script and print success message
def run_script(script_name):
    result = subprocess.run(["python", script_name])
    if result.returncode == 0:  # Check if the script executed successfully
        print(f"{GREEN}====== Successfully executed ====={script_name}{RESET}")
    else:
        print(f"{RED}===== Failed to execute ====={script_name}{RESET}")

# Run the scripts
print(f"{GREEN}=== Running Scripts ==={RESET}")
run_script("1.1.py")
run_script("1.2.py")
run_script("1.3.py")
run_script("2.py")
print(f"{GREEN}=== All Scripts Executed ==={RESET}")
