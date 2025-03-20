import subprocess

generated_script = "src/generated_graph.py"

try:
    subprocess.run(["python", generated_script], check=True)
    print(f"Executed {generated_script} successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing {generated_script}: {e}")