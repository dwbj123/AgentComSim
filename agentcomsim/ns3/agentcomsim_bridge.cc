// AgentComSim ns-3 packet-level bridge.
// Interface target: ns-3.43+ with C++17. See docs/ns3.md for validation scope.

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

namespace {

struct Message {
    std::string id;
    uint32_t taskId;
    std::string kind;
    double sendTime;
    uint32_t size;
};

std::vector<Message> g_messages;
std::map<uint32_t, uint32_t> g_received;
std::map<uint32_t, double> g_finished;
uint64_t g_packetCount = 0;
Ptr<Socket> g_sender;
Address g_destination;

std::vector<std::string> SplitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

std::vector<Message> ReadMessages(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open input CSV: " + path);
    std::string line;
    std::getline(input, line); // header
    std::vector<Message> messages;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto f = SplitCsv(line);
        if (f.size() != 5) throw std::runtime_error("Invalid input CSV row");
        messages.push_back({f[0], static_cast<uint32_t>(std::stoul(f[1])), f[2],
                            std::stod(f[3]), static_cast<uint32_t>(std::stoul(f[4]))});
    }
    return messages;
}

void Receive(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    while ((packet = socket->Recv())) {
        // Packet tags are local to a node. The four-byte wire prefix is
        // therefore part of the UDP payload and survives socket delivery.
        if (packet->GetSize() < 4) continue;
        uint8_t prefix[4];
        packet->CopyData(prefix, 4);
        const uint32_t id = (static_cast<uint32_t>(prefix[0]) << 24) |
                            (static_cast<uint32_t>(prefix[1]) << 16) |
                            (static_cast<uint32_t>(prefix[2]) << 8) |
                            static_cast<uint32_t>(prefix[3]);
        if (id >= g_messages.size()) continue;
        g_received[id] += packet->GetSize() - 4;
        g_packetCount++;
        if (g_received[id] >= g_messages[id].size && !g_finished.count(id)) {
            g_finished[id] = Simulator::Now().GetSeconds();
        }
    }
}

void SendMessage(uint32_t id) {
    const uint32_t payloadLimit = 1024;
    uint32_t sent = 0;
    while (sent < g_messages[id].size) {
        const uint32_t payload = std::min(payloadLimit, g_messages[id].size - sent);
        std::vector<uint8_t> wire(payload + 4, 0);
        wire[0] = static_cast<uint8_t>((id >> 24) & 0xff);
        wire[1] = static_cast<uint8_t>((id >> 16) & 0xff);
        wire[2] = static_cast<uint8_t>((id >> 8) & 0xff);
        wire[3] = static_cast<uint8_t>(id & 0xff);
        Ptr<Packet> packet = Create<Packet>(wire.data(), static_cast<uint32_t>(wire.size()));
        if (g_sender->SendTo(packet, 0, g_destination) < 0) {
            throw std::runtime_error("UDP send failed for message " + g_messages[id].id);
        }
        sent += payload;
    }
}

void WriteResults(const std::string& outputDir, double bandwidth, double delayMs) {
    std::filesystem::create_directories(outputDir);
    std::ofstream csv(outputDir + "/messages.csv");
    csv << "message_id,task_id,kind,send_time_s,size_bytes,finish_time_s,received_bytes\n";
    csv << std::fixed << std::setprecision(9);
    uint32_t completed = 0;
    for (uint32_t i = 0; i < g_messages.size(); ++i) {
        const auto& message = g_messages[i];
        const auto finish = g_finished.find(i);
        const double finishTime = finish == g_finished.end() ? -1.0 : finish->second;
        completed += finish != g_finished.end() ? 1 : 0;
        csv << message.id << ',' << message.taskId << ',' << message.kind << ','
            << message.sendTime << ',' << message.size << ',' << finishTime << ','
            << g_received[i] << '\n';
    }
    std::ofstream json(outputDir + "/summary.json");
    json << std::fixed << std::setprecision(6)
         << "{\n  \"message_count\": " << g_messages.size()
         << ",\n  \"completed_count\": " << completed
         << ",\n  \"packet_count\": " << g_packetCount
         << ",\n  \"bandwidth_mbps\": " << bandwidth
         << ",\n  \"delay_ms\": " << delayMs << "\n}\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string input;
    std::string output = "agentcomsim-ns3-results";
    double bandwidth = 25.0;
    double delayMs = 2.0;
    CommandLine cmd;
    cmd.AddValue("input", "Input message CSV", input);
    cmd.AddValue("output", "Output directory", output);
    cmd.AddValue("bandwidth", "Point-to-point bandwidth in Mbit/s", bandwidth);
    cmd.AddValue("delayMs", "Point-to-point propagation delay in ms", delayMs);
    cmd.Parse(argc, argv);
    if (input.empty() || bandwidth <= 0 || delayMs < 0) {
        std::cerr << "Usage: --input=messages.csv --output=dir --bandwidth=25 --delayMs=2\n";
        return 2;
    }

    g_messages = ReadMessages(input);
    NodeContainer nodes;
    nodes.Create(2);
    InternetStackHelper internet;
    internet.Install(nodes);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(static_cast<uint64_t>(bandwidth * 1e6))));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delayMs)));
    NetDeviceContainer devices = p2p.Install(nodes);
    Ipv4AddressHelper addresses;
    addresses.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = addresses.Assign(devices);

    Ptr<Socket> receiver = Socket::CreateSocket(nodes.Get(1), UdpSocketFactory::GetTypeId());
    receiver->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9000));
    receiver->SetRecvCallback(MakeCallback(&Receive));
    g_sender = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
    g_destination = InetSocketAddress(interfaces.GetAddress(1), 9000);

    double finalTime = 0.0;
    for (uint32_t i = 0; i < g_messages.size(); ++i) {
        Simulator::Schedule(Seconds(g_messages[i].sendTime), &SendMessage, i);
        finalTime = std::max(finalTime, g_messages[i].sendTime);
    }
    Simulator::Stop(Seconds(finalTime + 60.0));
    Simulator::Run();
    Simulator::Destroy();
    WriteResults(output, bandwidth, delayMs);
    return 0;
}
