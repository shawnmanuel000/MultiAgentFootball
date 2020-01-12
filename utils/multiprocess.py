import pickle
import numpy as np
import socket as Socket

class Worker():
	def __init__(self, self_port):
		self.port = self_port
		self.sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
		self.sock.setsockopt(Socket.SOL_SOCKET, Socket.SO_REUSEADDR, 1)
		self.sock.bind(("localhost", self_port))
		self.sock.listen(5)
		print(f"Worker listening on port {self_port} ...")
		self.conn = self.sock.accept()[0]
		print(f"Connected!")

	def start(self, gpu=True, iterations=1):
		raise NotImplementedError

	def __del__(self):
		self.conn.close()
		self.sock.close()

class Manager():
	def __init__(self, client_ports):
		self.num_clients = len(client_ports)
		self.client_ports = client_ports
		self.client_sockets = self.connect_sockets(client_ports)

	def start(self, popsize, epochs=1000):
		raise NotImplementedError

	def connect_sockets(self, ports):
		client_sockets = {port:None for port in ports}
		for port in ports:
			if client_sockets[port] is not None: continue
			try:
				sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
				sock.connect(("localhost", port))
				client_sockets[port] = sock
			except Exception:
				print("Couldn't connect to {}".format(port))
		return client_sockets

	def send_params(self, params, encoded=False):
		for p,port in zip(params, self.client_ports):
			# p = p if encoded else p.tostring()
			self.client_sockets[port].sendall(p)

	def await_results(self, converter=lambda x: x, decoded=False):
		responses = {}
		for port, sock in self.client_sockets.items():
			# data = sock.recv(100000)
			responses[port] = converter(sock.recv(100000))
		return [responses[port] for port in self.client_ports]

	def __del__(self):
		self.send_params([pickle.dumps({"cmd": "CLOSE", "item": [0.0]}) for _ in range(self.num_clients)], encoded=True)
		# for sock in self.client_sockets.values(): 
		# 	sock.sendall(np.zeros([], dtype=np.float64).tostring())
		for sock in self.client_sockets.values():
			sock.close()