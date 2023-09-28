import socket

# Set up UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('0.0.0.0', 5252)  # Replace with the server IP and port
sock.bind(server_address)

while True:
    # Receive the UDP packet
    data, address = sock.recvfrom(1024)

    # Print the received data
    print(f"Received: {data.decode()} from {address[0]}:{address[1]}")

    # Send a response (optional)
    response = "Message received!"
    sock.sendto(response.encode(), address)
