import pickle
import numpy as np
import socket as Socket
from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()

class TCPServer():
	def __init__(self, self_port):
		self.port = self_port
		self.sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
		self.sock.setsockopt(Socket.SOL_SOCKET, Socket.SO_REUSEADDR, 1)
		self.sock.bind(("localhost", self_port))
		self.sock.listen(5)
		print(f"Worker listening on port {self_port} ...")
		self.conn = self.sock.accept()[0]
		print(f"Connected!")

	def recv(self, decoder=pickle.loads):
		return decoder(self.conn.recv(1000000))

	def send(self, data, encoder=pickle.dumps):
		self.conn.sendall(encoder(data))

	def __del__(self):
		self.conn.close()
		self.sock.close()

class TCPClient():
	def __init__(self, client_ports):
		self.num_clients = len(client_ports)
		self.client_ports = client_ports
		self.client_sockets = self.connect_sockets(client_ports)

	def connect_sockets(self, ports):
		client_sockets = {port:None for port in ports}
		for port in ports:
			sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
			sock.connect(("localhost", port))
			client_sockets[port] = sock
		return client_sockets

	def broadcast(self, params, encoder=pickle.dumps):
		num = min(len(params), self.num_clients)
		[self.client_sockets[port].sendall(encoder(p)) for p,port in zip(params[:num], self.client_ports[:num])]
			
	def gather(self, decoder=pickle.loads):
		return [decoder(sock.recv(1000000)) for port, sock in self.client_sockets.items()]

	def __del__(self):
		for sock in self.client_sockets.values(): sock.close()

class MPIConnection():
	def __init__(self):
		self.rank = MPI_RANK

	def broadcast(self, params, encoder=lambda x: x):
		num = min(len(params), MPI_SIZE-1)
		[MPI_COMM.send(encoder(p), dest=i, tag=i) for p,i in zip(params[:num], range(1, num+1))]
			
	def gather(self, decoder=lambda x: x):
		return [decoder(MPI_COMM.recv(source=i, tag=i)) for i in range(1, MPI_SIZE)]

	def send(self, data, encoder=lambda x: x):
		MPI_COMM.send(encoder(data), dest=0, tag=self.rank)

	def recv(self, decoder=lambda x: x):
		return decoder(MPI_COMM.recv(source=0, tag=self.rank))
