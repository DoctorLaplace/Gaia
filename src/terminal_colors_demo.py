
"""
GAIA TERMINAL COLOR PALETTE DEMO
A quick utility to visualize ANSI escape codes for terminal styling.
"""

def print_palette():
    print("\n" + "="*50)
    print("      GAIA TERMINAL COLOR PALETTE OPTIONS")
    print("="*50 + "\n")

    # 1. Standard ANSI (30-37)
    print("--- STANDARD ANSI (30-37) ---")
    for i in range(30, 38):
        print(f"\033[{i}m Code {i} \033[0m", end=" ")
    print("\n")

    # 2. Bold/Bright ANSI (90-97)
    print("--- BOLD / BRIGHT ANSI (90-97) ---")
    for i in range(90, 98):
        # We also show them with the BOLD [1;] modifier
        print(f"\033[1;{i}m Code 1;{i} \033[0m", end=" ")
    print("\n")

    # 3. Gaia Project Official Colors
    print("--- GAIA PROJECT THEME COLORS ---")
    gaia_colors = {
        "PURPLE (Header)": "\033[1;95m",
        "GREEN (Success)": "\033[92m",
        "CYAN (Info)": "\033[96m",
        "RED (Fail)": "\033[91m",
        "BLUE (Alt Info)": "\033[94m",
        "CYAN (Standard)": "\033[36m",
        "WARNING (Yellow)": "\033[93m"
    }
    
    for name, code in gaia_colors.items():
        print(f"{code}{name:18} : {repr(code)}{'\033[0m'}")
    print("\n")

    # 4. Extended 256-Color Palette (Sample)
    print("--- EXTENDED XTERM-256 (Sample) ---")
    # Some extra vivid ones
    vivid_samples = [196, 201, 208, 118, 45, 226, 213]
    for code in vivid_samples:
        print(f"\033[38;5;{code}m Color {code:3} \033[0m", end=" ")
    print("\n")

    print("="*50)

if __name__ == "__main__":
    print_palette()
