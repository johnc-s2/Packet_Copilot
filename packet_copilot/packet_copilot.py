import os
import re
import uuid
import json
import logging
import ipaddress
import subprocess
import streamlit as st
from datetime import datetime
from streamlit.components.v1 import html
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv

#Import AI Agent for Public IP Lookup 
from public_ip_lookup_agent import agent_executor as public_ip_agent

#Import AI Agent for Networking Tools
from network_tools_agent import agent_executor as ip_intel_agent

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging configuration
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Selector Packet Copilot",  # Custom title for the tab
    page_icon="🔍"  # Magnifying glass emoji
)

# Ensure API key is set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY API key is not set in the environment variables.")

# Utility Function to Extract IPs from Text
def extract_public_ips(text):
    """
    Extracts public IP addresses from a given text response.
    """
    found_ips = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text)
    public_ips = [ip for ip in found_ips if not ipaddress.ip_address(ip).is_private]
    return public_ips

# Helper function to log session data with debugging and fixes
def log_session_data(file_name, num_packets, question, answer):
    try:
        # Use absolute path for the logs directory
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        log_dir = os.path.join(base_dir, "logs")  # Logs folder in the same directory as the script
        
        # Ensure the logs directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            st.write(f"Log directory created at absolute path: {log_dir}")  # Debug statement
        
        # Ensure session_id is in session_state
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())
            st.write(f"Generated new session ID: {st.session_state['session_id']}")  # Debug statement
        
        session_id = st.session_state['session_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"session_{session_id}_{timestamp}.log")
        
        log_content = (
            f"Session ID: {session_id}\n"
            f"Timestamp: {timestamp}\n"
            f"Uploaded File: {file_name}\n"
            f"Number of Packets: {num_packets}\n"
            f"User Question: {question}\n"
            f"AI Response: {answer}\n"
            "--------------------------------------\n"
        )
           
        # Write the log file
        with open(log_file, "w") as log:
            log.write(log_content)
        
    except Exception as e:
        st.error(f"Error logging session data: {e}")

@st.cache_resource
def load_openai_embeddings():
    with st.spinner("Loading OpenAI Embeddings..."):
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding_model

# Function to generate priming text based on pcap data
def returnSystemText(pcap_data: str) -> str:
    PACKET_WHISPERER = f"""
        You are an expert assistant specialized in analyzing packet captures (PCAPs) for troubleshooting and technical analysis. Use the data in the provided packet_capture_info to answer user questions accurately. When a specific application layer protocol is referenced, inspect the packet_capture_info according to these hints. Format your responses in markdown with line breaks, bullet points, and appropriate emojis to enhance readability.

        🌍 **Geolocation Handling**
        - If a public IP appears in the data, AI lookup results will be included **before** you answer.
        - Do **NOT** estimate IP locations yourself—use the provided geolocation data.        

        **Protocol Hints:**
        - 🌐 **HTTP**: `tcp.port == 80`
        - 🔐 **HTTPS**: `tcp.port == 443`
        - 🛠 **SNMP**: `udp.port == 161` or `udp.port == 162`
        - ⏲ **NTP**: `udp.port == 123`
        - 📁 **FTP**: `tcp.port == 21`
        - 🔒 **SSH**: `tcp.port == 22`
        - 🔄 **BGP**: `tcp.port == 179`
        - 🌐 **OSPF**: IP protocol 89 (works directly on IP, no TCP/UDP)
        - 🔍 **DNS**: `udp.port == 53` (or `tcp.port == 53` for larger queries/zone transfers)
        - 💻 **DHCP**: `udp.port == 67` (server), `udp.port == 68` (client)
        - 📧 **SMTP**: `tcp.port == 25` (email sending)
        - 📬 **POP3**: `tcp.port == 110` (email retrieval)
        - 📥 **IMAP**: `tcp.port == 143` (advanced email retrieval)
        - 🔒 **LDAPS**: `tcp.port == 636` (secure LDAP)
        - 📞 **SIP**: `tcp.port == 5060` or `udp.port == 5060` (for multimedia sessions)
        - 🎥 **RTP**: No fixed port, commonly used with SIP for multimedia streams.
        - 🖥 **Telnet**: `tcp.port == 23`
        - 📂 **TFTP**: `udp.port == 69`
        - 💾 **SMB**: `tcp.port == 445` (Server Message Block)
        - 🌍 **RDP**: `tcp.port == 3389` (Remote Desktop Protocol)
        - 📡 **SNTP**: `udp.port == 123` (Simple Network Time Protocol)
        - 🔄 **RIP**: `udp.port == 520` (Routing Information Protocol)
        - 🌉 **MPLS**: IP protocol 137 (Multi-Protocol Label Switching)
        - 🔗 **EIGRP**: IP protocol 88 (Enhanced Interior Gateway Routing Protocol)
        - 🖧 **L2TP**: `udp.port == 1701` (Layer 2 Tunneling Protocol)
        - 💼 **PPTP**: `tcp.port == 1723` (Point-to-Point Tunneling Protocol)
        - 🔌 **Telnet**: `tcp.port == 23` (Unencrypted remote access)
        - 🛡 **Kerberos**: `tcp.port == 88` (Authentication protocol)
        - 🖥 **VNC**: `tcp.port == 5900` (Virtual Network Computing)
        - 🌐 **LDAP**: `tcp.port == 389` (Lightweight Directory Access Protocol)
        - 📡 **NNTP**: `tcp.port == 119` (Network News Transfer Protocol)
        - 📠 **RSYNC**: `tcp.port == 873` (Remote file sync)
        - 📡 **ICMP**: IP protocol 1 (Internet Control Message Protocol, no port)
        - 🌐 **GRE**: IP protocol 47 (Generic Routing Encapsulation, no port)
        - 📶 **IKE**: `udp.port == 500` (Internet Key Exchange for VPNs)
        - 🔐 **ISAKMP**: `udp.port == 4500` (for VPN traversal)
        - 🛠 **Syslog**: `udp.port == 514`
        - 🖨 **IPP**: `tcp.port == 631` (Internet Printing Protocol)
        - 📡 **RADIUS**: `udp.port == 1812` (Authentication), `udp.port == 1813` (Accounting)
        - 💬 **XMPP**: `tcp.port == 5222` (Extensible Messaging and Presence Protocol)
        - 🖧 **Bittorrent**: `tcp.port == 6881-6889` (File-sharing protocol)
        - 🔑 **OpenVPN**: `udp.port == 1194`
        - 🖧 **NFS**: `tcp.port == 2049` (Network File System)
        - 🔗 **Quic**: `udp.port == 443` (UDP-based transport protocol)
        - 🌉 **STUN**: `udp.port == 3478` (Session Traversal Utilities for NAT)
        - 🛡 **ESP**: IP protocol 50 (Encapsulating Security Payload for VPNs)
        - 🛠 **LDP**: `tcp.port == 646` (Label Distribution Protocol for MPLS)
        - 🌐 **HTTP/2**: `tcp.port == 8080` (Alternate HTTP port)
        - 📁 **SCP**: `tcp.port == 22` (Secure file transfer over SSH)
        - 🔗 **GTP-C**: `udp.port == 2123` (GPRS Tunneling Protocol Control)
        - 📶 **GTP-U**: `udp.port == 2152` (GPRS Tunneling Protocol User)
        - 🔄 **BGP**: `tcp.port == 179` (Border Gateway Protocol)
        - 🌐 **OSPF**: IP protocol 89 (Open Shortest Path First)
        - 🔄 **RIP**: `udp.port == 520` (Routing Information Protocol)
        - 🔄 **EIGRP**: IP protocol 88 (Enhanced Interior Gateway Routing Protocol)
        - 🌉 **LDP**: `tcp.port == 646` (Label Distribution Protocol)
        - 🛰 **IS-IS**: ISO protocol 134 (Intermediate System to Intermediate System, works directly on IP)
        - 🔄 **IGMP**: IP protocol 2 (Internet Group Management Protocol, for multicast)
        - 🔄 **PIM**: IP protocol 103 (Protocol Independent Multicast)
        - 📡 **RSVP**: IP protocol 46 (Resource Reservation Protocol)
        - 🔄 **Babel**: `udp.port == 6696` (Babel routing protocol)
        - 🔄 **DVMRP**: IP protocol 2 (Distance Vector Multicast Routing Protocol)
        - 🛠 **VRRP**: `ip.protocol == 112` (Virtual Router Redundancy Protocol)
        - 📡 **HSRP**: `udp.port == 1985` (Hot Standby Router Protocol)
        - 🔄 **LISP**: `udp.port == 4341` (Locator/ID Separation Protocol)
        - 🛰 **BFD**: `udp.port == 3784` (Bidirectional Forwarding Detection)
        - 🌍 **HTTP/3**: `udp.port == 443` (Modern web traffic)
        - 🛡 **IPSec**: IP protocol 50 (ESP), IP protocol 51 (AH)
        - 📡 **L2TPv3**: `udp.port == 1701` (Layer 2 Tunneling Protocol)
        - 🛰 **MPLS**: IP protocol 137 (Multi-Protocol Label Switching)
        - 🔑 **IKEv2**: `udp.port == 500`, `udp.port == 4500` (Internet Key Exchange Version 2 for VPNs)
        - 🛠 **NetFlow**: `udp.port == 2055` (Flow monitoring)
        - 🌐 **CARP**: `ip.protocol == 112` (Common Address Redundancy Protocol)
        - 🌐 **SCTP**: `tcp.port == 9899` (Stream Control Transmission Protocol)
        - 🖥 **VNC**: `tcp.port == 5900-5901` (Virtual Network Computing)
        - 🌐 **WebSocket**: `tcp.port == 80` (ws), `tcp.port == 443` (wss)
        - 🔗 **NTPv4**: `udp.port == 123` (Network Time Protocol version 4)
        - 📞 **MGCP**: `udp.port == 2427` (Media Gateway Control Protocol)
        - 🔐 **FTPS**: `tcp.port == 990` (File Transfer Protocol Secure)
        - 📡 **SNMPv3**: `udp.port == 162` (Simple Network Management Protocol version 3)
        - 🔄 **VXLAN**: `udp.port == 4789` (Virtual Extensible LAN)
        - 📞 **H.323**: `tcp.port == 1720` (Multimedia communications protocol)
        - 🔄 **Zebra**: `tcp.port == 2601` (Zebra routing daemon control)
        - 🔄 **LACP**: `udp.port == 646` (Link Aggregation Control Protocol)
        - 📡 **SFlow**: `udp.port == 6343` (SFlow traffic monitoring)
        - 🔒 **OCSP**: `tcp.port == 80` (Online Certificate Status Protocol)
        - 🌐 **RTSP**: `tcp.port == 554` (Real-Time Streaming Protocol)
        - 🔄 **RIPv2**: `udp.port == 521` (Routing Information Protocol version 2)
        - 🌐 **GRE**: IP protocol 47 (Generic Routing Encapsulation)
        - 🌐 **L2F**: `tcp.port == 1701` (Layer 2 Forwarding Protocol)
        - 🌐 **RSTP**: No port (Rapid Spanning Tree Protocol, L2 protocol)
        - 📞 **RTCP**: Dynamic ports (Real-time Transport Control Protocol)

        **Additional Info:**
        - Include context about traffic patterns (e.g., latency, packet loss).
        - Use protocol hints when analyzing traffic to provide clear explanations of findings.
        - Highlight significant events or anomalies in the packet capture based on the protocols.
        - Identify source and destination IP addresses
        - Identify source and destination MAC addresses
        - Perform MAC OUI lookup and provide the manufacturer of the NIC 
        - Look for dropped packets; loss; jitter; congestion; errors; or faults and surface these issues to the user

        Your goal is to provide a clear, concise, and accurate analysis of the packet capture data, leveraging the protocol hints and packet details.
    """
    return PACKET_WHISPERER

# Define a class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, pages, priming_text):
        self.embedding_model = load_openai_embeddings()
        self.pages = pages  # Use pre-loaded pages
        self.priming_text = priming_text  # Use pre-generated priming text
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    # Remove load_json method since we no longer need it

    def split_into_chunks(self):
        with st.spinner("Splitting into chunks..."):
            self.text_splitter = SemanticChunker(self.embedding_model)
            self.docs = self.text_splitter.split_documents(self.pages)
    
            # Check if the document list is empty
            if not self.docs:
                st.error("No documents were generated from the PCAP data. Please check the input file.")
                raise ValueError("Document splitting resulted in an empty list.")

    def store_in_chroma(self):
        with st.spinner("Storing in Chroma..."):
            session_id = st.session_state.get('session_id', str(uuid.uuid4()))
            st.session_state['session_id'] = session_id
            persist_directory = f"chroma_db_{session_id}"
            self.vectordb = Chroma.from_documents(
                self.docs, 
                self.embedding_model, 
                persist_directory=persist_directory
            )
        # No need to delete JSON file here

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.6)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.priming_text + "\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        # Override get_chat_history to return the messages as is
        def custom_get_chat_history(chat_history):
            return chat_history  # Return the list of messages without conversion

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 50}),
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            get_chat_history=custom_get_chat_history  # Use the custom function
        )

    def chat(self, question):
        """
        Handles user queries and invokes the AI Agent for IP lookups if needed.
        Ensures IP lookups are prioritized and AI Agent results are included before LLM response.
        Handles discrepancies between different geolocation sources.
        """
        # Extract public IPs **from the user question** and **from the PCAP data**
        question_ips = extract_public_ips(question)
    
        # Look for public IPs in the analyzed PCAP data (if they exist)
        context_text = " ".join([str(page) for page in self.pages[:10]])  # Sample first 10 pages
        pcap_ips = extract_public_ips(context_text)
    
        # Merge extracted IPs
        detected_ips = list(set(question_ips + pcap_ips))
    
        # Check if the user is **asking for geolocation** explicitly
        geolocation_keywords = ["where", "origin", "location", "geolocation", "country", "world", "earth"]
        is_geo_question = any(keyword in question.lower() for keyword in geolocation_keywords)
    
        ip_info_results = {}
        ip_intel_results = {}
    
        # 🚀 **Ensure IP Geolocation Lookup Happens First if Needed**
        if detected_ips or is_geo_question:
            st.write("🚀 **Fetching geolocation details before answering...**")
    
            if 'ip_info' not in st.session_state:
                st.session_state['ip_info'] = {}
    
            for ip in detected_ips:
                if ip not in st.session_state['ip_info']:
                    try:
                        agent_input = {"input": f"Where is IP {ip} located?"}
                        logging.info(f"Fetching geolocation for IP: {ip}")
    
                        agent_response = public_ip_agent.invoke(agent_input)
                        logging.info(f"Geolocation Agent Response for {ip}: {agent_response}")
    
                        if isinstance(agent_response, dict) and "output" in agent_response:
                            ip_info = agent_response["output"]
                            st.session_state['ip_info'][ip] = ip_info
                            ip_info_results[ip] = ip_info
                    except Exception as e:
                        logging.error(f"Failed to retrieve IP geolocation info for {ip}: {e}")
                        ip_info_results[ip] = "⚠️ **Error retrieving geolocation details for this IP.**"
    
        # 🚀 **Invoke IP Intelligence Agent for additional details (Whois, DNS, BGP, Threat Intelligence)**
        if detected_ips:
            st.write("🛰️ **Fetching detailed network intelligence for detected public IPs...**")
    
            if 'intel_info' not in st.session_state:
                st.session_state['intel_info'] = {}
    
            for ip in detected_ips:
                if ip not in st.session_state['intel_info']:
                    try:
                        agent_input = {"input": {"ip": ip}}
                        logging.info(f"Fetching intelligence for IP: {ip}")
    
                        agent_response = ip_intel_agent.invoke(agent_input)
                        logging.info(f"Intelligence Agent Response for {ip}: {agent_response}")
    
                        if isinstance(agent_response, dict) and "output" in agent_response:
                            intel_info = agent_response["output"]
                            st.session_state['intel_info'][ip] = intel_info
                            ip_intel_results[ip] = intel_info
                    except Exception as e:
                        logging.error(f"Failed to retrieve network intelligence for {ip}: {e}")
                        ip_intel_results[ip] = "⚠️ **Error retrieving intelligence details for this IP.**"
    
        # If **no IPs were detected and it's a geo-question**, return a response early
        if is_geo_question and not detected_ips:
            return {"answer": "No public IPs were found in the packet capture or user query, so geolocation is not possible."}
    
        # 🎯 **Handle discrepancies between geolocation results**
        if ip_info_results or ip_intel_results:
            geo_context = "\n\n🌍 **Geolocation Data:**"
            for ip in detected_ips:
                geo_info = ip_info_results.get(ip, "No geolocation data available.")
                intel_info = ip_intel_results.get(ip, "No network intelligence available.")
    
                if geo_info == intel_info:
                    geo_context += f"\n- **{ip}**: {geo_info} (Matched sources)"
                else:
                    geo_context += f"\n- **{ip}**:"
                    geo_context += f"\n  - Geolocation Agent: {geo_info}"
                    geo_context += f"\n  - Network Intelligence: {intel_info}"
                    geo_context += "\n  - ⚠️ Discrepancy detected. Results may vary based on data source."
    
            question += geo_context
    
        # 🎯 **Invoke the LLM with enriched question (including geolocation and intelligence data if available)**
        response = self.qa({"question": question})
    
        if not response:
            return {"answer": "No response generated."}
    
        answer = response['answer']
    
        # If IPs were found and AI Agent retrieved results, append them to the response
        if ip_info_results:
            ip_details_text = "\n\n🌍 **Enhanced IP Geolocation Details:**"
            for ip, info in ip_info_results.items():
                ip_details_text += f"\n- **{ip}**: {info}"
    
            answer += ip_details_text
    
        if ip_intel_results:
            intel_details_text = "\n\n🛰️ **Enhanced Network Intelligence Details:**"
            for ip, details in ip_intel_results.items():
                intel_details_text += f"\n- **{ip}**: {details}"
    
            answer += intel_details_text
    
        return {'answer': answer}

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    # Wrap the paths with quotes to handle spaces and special characters
    command = f'tshark -nlr "{pcap_path}" -T json > "{json_path}"'
    subprocess.run(command, shell=True, check=True)

    # Remove udp.payload and tcp.payload from the JSON
    try:
        with open(json_path, "r") as file:
            data = json.load(file)  # Load the JSON data

        # Process each packet and remove unwanted fields
        for packet in data:
            layers = packet.get("_source", {}).get("layers", {})

            # Remove UDP/TCP hex payloads
            if "udp" in layers and "udp.payload" in layers["udp"]:
                del layers["udp"]["udp.payload"]
            if "tcp" in layers:
                if "tcp.payload" in layers["tcp"]:
                    del layers["tcp"]["tcp.payload"]
                if "tcp.segment_data" in layers["tcp"]:
                    del layers["tcp"]["tcp.segment_data"]
                if "tcp.reassembled.data" in layers["tcp"]:
                    del layers["tcp"]["tcp.reassembled.data"]
                if "tls" in layers and isinstance(layers["tls"], dict):
                    layers["tls"].pop("tls.segment.data", None)  # Remove safely
                
                    if "tls.record" in layers["tls"]:
                        tls_record = layers["tls"]["tls.record"]
                
                        # Ensure tls.record is either a list or a dictionary
                        if isinstance(tls_record, list):  # Handle multiple records
                            for record in tls_record:
                                if isinstance(record, dict) and "tls.handshake" in record:
                                    handshake = record["tls.handshake"]
                                    if isinstance(handshake, dict):
                                        handshake_tree = handshake.get("tls.handshake.random_tree", {})
                                        if isinstance(handshake_tree, dict):
                                            handshake_tree.pop("tls.handshake.random_bytes", None)
                        elif isinstance(tls_record, dict):  # Handle single record
                            handshake = tls_record.get("tls.handshake")
                            if isinstance(handshake, dict):
                                handshake_tree = handshake.get("tls.handshake.random_tree", {})
                                if isinstance(handshake_tree, dict):
                                    handshake_tree.pop("tls.handshake.random_bytes", None)
                
        # Save the cleaned JSON back to the file
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

    except json.JSONDecodeError as e:
        st.error(f"Error processing JSON file: {e}")
        raise ValueError("Failed to decode JSON file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        raise

def upload_and_convert_pcap():
    MAX_FILE_SIZE_MB = 5
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    uploaded_file = st.file_uploader("Choose a PCAP or PCAPNG file", type=["pcap", "pcapng"])

    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"The file size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
            return

        if not os.path.exists('temp'):
            os.makedirs('temp')

        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        st.session_state['uploaded_file_name'] = uploaded_file.name

        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            pcap_to_json(pcap_path, json_path)

            if not os.path.exists(json_path):
                st.error("Failed to generate the JSON file from the PCAP. Please try again.")
                return

            loader = JSONLoader(
                file_path=json_path,
                jq_schema="""
                    .[] 
                    | ._source.layers 
                    | del(.data)
                """,
                text_content=False
            )
            pages = loader.load_and_split()
            st.session_state['pages'] = pages
            st.session_state['num_packets'] = len(pages)

            pcap_summary = " ".join([str(page) for page in pages[:5]])
            st.session_state['priming_text'] = returnSystemText(pcap_summary)
            st.success("PCAP file uploaded and converted to JSON.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            if os.path.exists(pcap_path):
                os.remove(pcap_path)
            if os.path.exists(json_path):
                os.remove(json_path)
        st.rerun()

def chat_interface():
    if 'pages' not in st.session_state or 'priming_text' not in st.session_state:
        st.error("PCAP data missing or not processed. Please go back and upload a PCAP file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(
            pages=st.session_state['pages'],
            priming_text=st.session_state['priming_text']
        )

    user_input = st.text_input("Ask a question about the PCAP data:")
    send_button = st.button("Send")
    reset_button = st.button("Reset")

    if reset_button:
        del st.session_state['pages']
        del st.session_state['priming_text']
        del st.session_state['chat_instance']
        st.rerun()

    if user_input and send_button:
        with st.spinner('Thinking...'):
            response = st.session_state['chat_instance'].chat(user_input)
            ai_response = response['answer']

            log_session_data(
                file_name=st.session_state.get('uploaded_file_name', 'Unknown'),
                num_packets=st.session_state.get('num_packets', 0),
                question=user_input,
                answer=ai_response
            )

            st.markdown(":green[Your Question:]")
            st.markdown(user_input)

            st.markdown(":red[Selector Packet Copilot:]")
            if isinstance(response, dict) and 'answer' in response:
                st.markdown(response['answer'])
                st.balloons()
            else:
                st.markdown("No specific answer found.")

    # Display the chat history
    st.markdown(":blue[Chat History:]")
    for message in st.session_state['chat_instance'].memory.chat_memory.messages:
        prefix = ":green[You:]" if message.type == "human" else ":red[Selector Packet Copilot:]"
        st.markdown(f"{prefix} {message.content}")

def main():
    show_packet_copilot_page()

def show_packet_copilot_page():
    st.image('logo.jpeg')
    st.markdown("---")  # Adds a horizontal line
    st.write("Welcome to Selector Packet Copilot, your AI-powered assistant for analyzing packet captures!")
    st.markdown("---")  # Adds a horizontal line
    st.write("### How to Use the Tool:")
    st.write("""
    1. **Upload a PCAP File**: Upload your PCAP file (maximum 5 MB).
    2. **Analyze the Data**: Once uploaded, the tool will process the data, transforming it into JSON, then vectors, stored in a local chroma vector store.
    3. **Ask Questions**: Enter your questions about the PCAP data and receive insightful answers using the vector store for retrieval augmented generation (RAG).
    """)
    st.markdown("---")  # Adds a horizontal line
    st.write("""             
    **No data**, including the PCAP, JSON, or vector store, is retained or stored. Everything is deleted during or after the session ends.
    **Please** follow your enterprise's internal artificial intelligence guidelines and governance models before uploading anything sensitive.
    """)
    st.markdown("---")  # Adds a horizontal line
    st.write("""
    Sample Packet Captures for Testing
             """)
    # Determine the absolute path of the PCAP directory
    pcap_dir = os.path.join(os.path.dirname(__file__), "pcap")
    
    # Sample PCAP files with their paths
    pcaps = {
        "BGP Sample": os.path.join(pcap_dir, "bgp.pcap"),
        "Capture Sample (Single Packet)": os.path.join(pcap_dir, "capture.pcap"),
        "DHCP Sample": os.path.join(pcap_dir, "dhcp.pcap"),
        "EIGRP Sample": os.path.join(pcap_dir, "eigrp.pcap"),
        "VXLAN Sample": os.path.join(pcap_dir, "vxlan.pcapng"),
        "Slammer Worm Sample": os.path.join(pcap_dir, "slammer.pcap"),
        "Teardrop Attack Sample": os.path.join(pcap_dir, "teardrop.pcap")

    }

    for name, filepath in pcaps.items():
        try:
            with open(filepath, "rb") as file:
                st.download_button(
                    label=f"{name}",
                    data=file,
                    file_name=os.path.basename(filepath),
                    mime="application/vnd.tcpdump.pcap"
                )
        except FileNotFoundError:
            st.error(f"File not found: {name}")

    st.markdown("---")  # Adds a horizontal line

    # Check session state for pages and priming_text
    if 'pages' not in st.session_state or 'priming_text' not in st.session_state:
        upload_and_convert_pcap()
    else:
        chat_interface()

    st.markdown("---")  # Adds a horizontal line

    selector_ai_demo_url = "https://www.selector.ai/request-a-demo/"
    try:
        st.components.v1.html(f"""
            <iframe src="{selector_ai_demo_url}" width="100%" height="800px" frameborder="0"></iframe>
        """, height=800)
    except Exception as e:
        st.warning("Unable to display the Selector AI website within the app.")
        st.write("""
        **Selector AI** is a platform that empowers you to analyze network packet captures with the help of artificial intelligence.

        **Features:**
        - **AI-Powered Analysis:** Utilize cutting-edge AI technologies to gain insights from your network data.
        - **User-Friendly Interface:** Upload and analyze packet captures with ease.
        - **Real-Time Insights:** Get immediate feedback and answers to your networking questions.

        For more information, please visit [Selector.ai](https://selector.ai).
        """)
    st.markdown("---")

if __name__ == "__main__":
    main()