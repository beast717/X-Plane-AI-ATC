import socket
import struct

class XPlaneConnect(object):
    '''XPlaneConnect (XPC) facilitates communication to and from the XPCPlugin.'''
    socket = None

    def __init__(self, xpHost = 'localhost', xpPort = 49009, port = 0, timeout = 100):
        xpIP = None
        try:
            xpIP = socket.gethostbyname(xpHost)
        except:
            raise ValueError("Unable to resolve xpHost.")

        self.xpDst = (xpIP, xpPort)
        clientAddr = ("0.0.0.0", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.bind(clientAddr)
        self.socket.settimeout(timeout / 1000.0)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def sendUDP(self, buffer):
        if(len(buffer) == 0):
            raise ValueError("sendUDP: buffer is empty.")
        # FIX: Ensure we are sending bytes
        if isinstance(buffer, str):
            buffer = buffer.encode('utf-8')
        self.socket.sendto(buffer, 0, self.xpDst)

    def readUDP(self):
        return self.socket.recv(16384)

    def getPOSI(self, ac = 0):
        # FIX: Encode the header "GETP"
        buffer = struct.pack("<4sxB", "GETP".encode(), ac)
        self.sendUDP(buffer)

        resultBuf = self.readUDP()
        if len(resultBuf) != 34:
            raise ValueError("Unexpected response length.")

        result = struct.unpack("<4sxBfffffff", resultBuf)
        # FIX: Decode the header for comparison
        header = result[0].decode('utf-8').strip('\x00')
        if header != "POSI":
            raise ValueError("Unexpected header: " + header)

        return result[2:]

    def getDREFs(self, drefs):
        # FIX: Encode "GETD"
        buffer = struct.pack("<4sxB", "GETD".encode(), len(drefs))
        for dref in drefs:
            # FIX: Encode the DREF string
            fmt = "<B{0:d}s".format(len(dref))
            buffer += struct.pack(fmt, len(dref), dref.encode())
        self.sendUDP(buffer)

        buffer = self.readUDP()
        resultCount = struct.unpack_from("B", buffer, 5)[0]
        offset = 6
        result = []
        for i in range(resultCount):
            rowLen = struct.unpack_from("B", buffer, offset)[0]
            offset += 1
            fmt = "<{0:d}f".format(rowLen)
            row = struct.unpack_from(fmt, buffer, offset)
            result.append(row)
            offset += rowLen * 4
        return result

    def getDREF(self, dref):
        return self.getDREFs([dref])[0]