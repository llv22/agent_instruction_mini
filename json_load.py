import json

if __name__ == "__main__":
    with open("request.json", "r") as f:
        data = f.read()
    print(data[1619:1630])