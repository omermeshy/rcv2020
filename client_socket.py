import socket
import errno
from video_from_the_robot import get_data, init
import logging


def send_data():
    global connection_available
    logging.debug("sending data")
    data = get_data().encode('utf-8')
    try:
        client_socket.sendall(data)
        logging.debug("sent successfully")
    except socket.error as e:
        logging.error("error in sending " + str(e))
        client_socket.close()
        connection_available = False


def init_socket():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(1)


def establish_connection():
    logging.debug("trying to connect")
    init_socket()
    err = client_socket.connect_ex((ip, port))
    if err == socket.error:
        pass
    if err != 0:
        logging.error("couldn't connect " + str(err))
        client_socket.close()
        return False
    else:
        init()
        logging.debug("connected")
        return True


def connect_and_send():
    global connection_available
    if not connection_available:
        connection_available = establish_connection()
    if connection_available:
        send_data()


def main():
    global ip
    global port
    global connection_available
    ip = '10.56.35.2'
    port = 5801
    connection_available = False
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/home/pi/rcv/logs/socket_client.log',
                        filemode='w')
    while True:
        connect_and_send()


if __name__ == '__main__':
    main()
