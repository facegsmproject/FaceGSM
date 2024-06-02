import sys

def show_error(message, exit=True):
    print("[!] Error: ", message)
    if exit:
        sys.exit()


def show_error_arg(message, arg):
    print(f"[!] Error: {message} for {arg} argument")
    sys.exit()


def show_info(message):
    print(f"[+] {message}")
