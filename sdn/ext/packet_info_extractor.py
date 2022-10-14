import pox.lib.packet as pkt

def handle_TCP_packets (tcp_packet, packet_info):
    # extract tcp level data
    packet_info["srcport"] = tcp_packet.srcport
    packet_info["dstport"] = tcp_packet.dstport
    packet_info["seq"] = tcp_packet.seq
    packet_info["ack"] = tcp_packet.ack
    packet_info["off"] = tcp_packet.off
    packet_info["win"] = tcp_packet.win
    packet_info["tcplen"] = tcp_packet.tcplen
    packet_info["SYN"] = tcp_packet.SYN
    packet_info["ACK"] = tcp_packet.ACK
    packet_info["FIN"] = tcp_packet.FIN
    packet_info["RST"] = tcp_packet.RST
    packet_info["PSH"] = tcp_packet.PSH
    packet_info["URG"] = tcp_packet.URG
    packet_info["ECN"] = tcp_packet.ECN
    packet_info["CWR"] = tcp_packet.CWR

def handle_IP_packet (ip_packet):
    packet_info = {}

    # extract ip level data
    packet_info["srcip"] = ip_packet.srcip
    packet_info["dstip"] = ip_packet.dstip
    packet_info["ttl"] = ip_packet.ttl
    packet_info["iplen"] = ip_packet.iplen

    if ip_packet.protocol == pkt.ipv4.TCP_PROTOCOL:
        handle_TCP_packets(ip_packet.payload, packet_info)
    
    return packet_info