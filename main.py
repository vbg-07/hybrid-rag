import sys
import subprocess

def clear_screen():
    subprocess.run(["clear" if sys.platform != "win32" else "cls"], shell=True)

def run_script(script_name):
    """Runs a python script in an isolated process to free up memory afterward."""
    try:
        print(f"\n Launching {script_name}...\n")
        subprocess.run([sys.executable, script_name])
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{script_name}'.")
        print("Please ensure it is in the same folder as main.py.")
    except Exception as e:
        print(f"\n❌ An error occurred while running {script_name}: {e}")
    
    input("\nPress Enter to return to the main menu...")

def main():
    while True:
        clear_screen()
        print("=" * 45)
        print("  LOCAL RAG PIPELINE CONTROL CENTER")
        print("=" * 45)
        print("  1.  Ask a Question (Query)")
        print("  2.  Add New Documents (Ingest/Update)")
        print("  3.  Delete a Document (Prune DB)")
        print("  4.  Exit")
        print("=" * 45)
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            run_script("query.py")
        elif choice == '2':
            run_script("ingest.py")
        elif choice == '3':
            run_script("delete.py") 
        elif choice == '4':
            clear_screen()
            print(" Shutting down RAG Pipeline. Goodbye!")
            break
        else:
            print("\n⚠️ Invalid selection. Please choose a number between 1 and 4.")
            input("Press Enter to try again...")

if __name__ == "__main__":
    main()