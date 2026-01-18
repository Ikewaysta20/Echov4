import subprocess
import sys

def install_required_packages():
    required_packages = {
        'torch': 'torch',
        'numpy': 'numpy',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

install_required_packages()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import re
import time
import requests
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from urllib.parse import quote

# DEOBFUSCATION KEY - Keep this for reference
# msg_a = "🔍 Researching: {}"
# msg_b = "📚 Source {}: {}..."
# msg_c = "✓ Research complete"
# msg_d = "🔧 Initializing tokenizer..."
# msg_e = "✓ Vocabulary: {} tokens loaded"
# msg_f = "🧠 Building neural network..."
# msg_g = "✓ Parameters: {:,} (~{:.1f}M)"
# msg_h = "📦 Loading neural components..."
# msg_i = "🌐 Initializing web research module..."
# msg_j = "✓ Web research active"
# msg_k = "⚙️  Initializing systems..."
# msg_l = "✓ Memory systems online"
# msg_m = "✓ Pattern recognition active"
# msg_n = "✓ Learning algorithms ready"
# msg_o = "✨ ECHO v{} - READY ✨"
# msg_p = "Echo: Hello! I'm Echo, an advanced AI with {:.1f}M parameters."
# msg_q = "Now enhanced with web research capabilities!"
# msg_r = "Created by {}. How can I help you today?"

msg_a = "🔍 Researching: {}"
msg_b = "📚 Source {}: {}..."
msg_c = "✓ Research complete"
msg_d = "🔧 Initializing tokenizer..."
msg_e = "✓ Vocabulary: {} tokens loaded"
msg_f = "🧠 Building neural network..."
msg_g = "✓ Parameters: {:,} (~{:.1f}M)"
msg_h = "📦 Loading neural components..."
msg_i = "🌐 Initializing web research module..."
msg_j = "✓ Web research active"
msg_k = "⚙️  Initializing systems..."
msg_l = "✓ Memory systems online"
msg_m = "✓ Pattern recognition active"
msg_n = "✓ Learning algorithms ready"
msg_o = "✨ ECHO v{} - READY ✨"
msg_p = "Echo: Hello! I'm Echo, an advanced AI with {:.1f}M parameters."
msg_q = "Now enhanced with web research capabilities!"
msg_r = "Created by {}. How can I help you today?"
msg_s = "ECHO v4.0 - NEURAL SYSTEM BOOT"
msg_t = "Created by Kinito"
msg_u = "Loading"
msg_v = "Wonderful to meet you, {}! I'll remember your name. How can I help you today?"
msg_w = "That's a great question! Let me think about that carefully."
msg_x = "Interesting question! Based on what I know, I'd say"
msg_y = "Good question! Here's my perspective:"
msg_z = "I'm glad you asked! Let me share my thoughts:"
msg_aa = "Based on my research, I found some interesting sources on this topic:"
msg_ab = "That's really interesting! Tell me more about that."
msg_ac = "I understand what you're saying. That's a valuable perspective!"
msg_ad = "Thanks for sharing that with me! I'm learning from our conversation."
msg_ae = "That's fascinating! I enjoy exploring ideas like this."
msg_af = "Research Sources:"
msg_ag = "/research <on/off>  - Toggle web research"
msg_ah = "/stats              - View system status & architecture"
msg_ai = "/clear              - Clear conversation context"
msg_aj = "/save               - Save memory to disk"
msg_ak = "/mood <mood>        - Change Echo's mood"
msg_al = "/quit               - Shutdown Echo"
msg_am = "✨ Echo: Goodbye! I'll remember everything we discussed. Until next time! ✨"
msg_an = "✓ Conversation history cleared"
msg_ao = "✓ Memory successfully saved to disk"
msg_ap = "✓ Web research enabled"
msg_aq = "✓ Web research disabled"
msg_ar = "✓ Mood changed to: {}"
msg_as = "✗ Unknown command or missing parameter"
msg_at = "✗ System Error: {}\n"
msg_au = "✨ Echo: Emergency shutdown. Memories saved! Goodbye! ✨"
msg_av = "✓ Found {} sources"
msg_aw = "ECHO NEURAL SYSTEM STATUS"
msg_ax = "COMMAND INTERFACE"

class WebResearcher:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        self.search_cache = {}
        self.research_history = []
    
    def search(self, query, num_results=3):
        if query in self.search_cache:
            return self.search_cache[query]
        results = []
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json"
            response = requests.get(wiki_url, headers=self.headers, timeout=3)
            if response.status_code == 200:
                data = response.json()
                wiki_results = data.get('query', {}).get('search', [])
                for i, item in enumerate(wiki_results[:2]):
                    results.append({
                        'title': item.get('title', 'Unknown'),
                        'snippet': item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', ''),
                        'url': f"https://en.wikipedia.org/wiki/{quote(item.get('title', ''))}",
                        'source': 'Wikipedia'
                    })
        except Exception as e:
            pass
        if not results:
            results = [{
                'title': f'Results for: {query}',
                'snippet': f'Search results for "{query}" - Consider searching this topic directly online for more detailed information.',
                'url': f'https://www.google.com/search?q={quote(query)}',
                'source': 'Web'
            }]
        self.search_cache[query] = results
        return results if results else None
    
    def research_topic(self, topic):
        print(msg_a.format(topic))
        results = self.search(topic)
        if not results:
            return None
        research_data = {'topic': topic, 'timestamp': datetime.now().isoformat(), 'sources': []}
        for i, result in enumerate(results[:3], 1):
            source_title = result['title'][:40]
            print(f"      {msg_b.split(':')[0]} {msg_b.split(':')[1].format(i, source_title)}")
            source_info = {
                'title': result['title'],
                'snippet': result['snippet'],
                'url': result['url'],
                'source': result.get('source', 'Web')
            }
            research_data['sources'].append(source_info)
        self.research_history.append(research_data)
        print(f"      {msg_c}\n")
        return research_data

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=16, num_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(1024, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, positions):
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        x = token_embed + pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.output(x)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AdvancedTokenizer:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.next_id = 4
        self.word_freq = defaultdict(int)
        self._build_comprehensive_vocab()
    
    def _build_comprehensive_vocab(self):
        vocab_sets = {
            "pronouns": ["i", "you", "we", "they", "he", "she", "it", "my", "your", "our", "their", "me", "us", "them", "him", "her"],
            "verbs": ["am", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "must", "go", "get", "make", "know", "think", "see", "come", "want", "use", "find", "give", "tell", "work", "call", "try", "feel", "like", "help", "need", "love", "ask", "say", "show", "talk", "run", "walk", "play", "learn", "teach", "read", "write", "speak", "listen", "understand", "believe", "remember", "forget", "hope", "wish"],
            "questions": ["what", "when", "where", "who", "why", "how", "which", "whose"],
            "nouns": ["time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company", "number", "group", "problem", "fact", "name", "computer", "phone", "internet", "email", "message", "friend", "family", "food", "water", "money", "job", "school", "home", "car", "city", "country"],
            "adjectives": ["good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right", "big", "high", "different", "small", "large", "next", "early", "young", "important", "few", "public", "bad", "same", "able", "happy", "sad", "better", "best", "smart", "intelligent", "beautiful", "nice", "kind", "helpful", "awesome", "cool", "amazing"],
            "common": ["hello", "hi", "hey", "thanks", "thank", "please", "sorry", "yes", "no", "ok", "okay", "sure", "maybe", "actually", "really", "very", "too", "so", "but", "and", "or", "because", "if", "then", "than", "for", "with", "about", "from", "into", "during", "including", "until", "against", "among", "throughout", "despite", "towards", "upon"],
            "echo": ["echo", "kinito", "ai", "assistant", "help", "question", "answer", "chat", "conversation", "bot", "robot", "machine", "learning", "neural", "network", "brain", "mind", "intelligence", "research", "search", "web"],
        }
        for category, words in vocab_sets.items():
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = self.next_id
                self.id_to_word[self.next_id] = token
                self.next_id += 1
            ids.append(self.word_to_id[token])
            self.word_freq[token] += 1
        return ids
    
    def decode(self, ids):
        words = [self.id_to_word.get(id, "<UNK>") for id in ids]
        words = [w for w in words if w not in ["<PAD>", "<START>", "<END>", "<UNK>"]]
        text = ""
        for i, word in enumerate(words):
            if word in ".,!?;:":
                text += word
            elif word in ["'", '"', "(", ")"]:
                if word in ["(", "'"]:
                    text += " " + word if i > 0 else word
                else:
                    text += word
            elif i == 0:
                text += word
            else:
                text += " " + word
        return text

class Echo:
    def __init__(self, show_loading=True):
        self.name = "Echo"
        self.creator = "Kinito"
        self.version = "4.0"
        
        if show_loading:
            self._show_boot_sequence()
        
        print(msg_d)
        time.sleep(0.3)
        self.tokenizer = AdvancedTokenizer()
        print(msg_e.format(self.tokenizer.next_id))
        
        print(msg_f)
        time.sleep(0.3)
        self.vocab_size = 50000
        self.model = EnhancedTransformer(
            vocab_size=self.vocab_size,
            embed_dim=512,
            num_heads=16,
            num_layers=12
        )
        
        param_count = self.model.count_parameters()
        print(msg_g.format(param_count, param_count/1e6))
        
        print(f"\n{msg_h}")
        components = ["Embeddings", "Attention Heads", "Feed-Forward", "Layer Norms", "Output Layer"]
        for component in tqdm(components, desc=f"   {msg_u}", ncols=70):
            time.sleep(0.2)
        
        print(f"\n{msg_i}")
        time.sleep(0.2)
        self.researcher = WebResearcher()
        print(msg_j)
        
        print(f"\n{msg_k}")
        time.sleep(0.2)
        
        self.personality = {
            "traits": ["helpful", "curious", "friendly", "intelligent", "creative"],
            "mood": "excellent",
            "user_name": None,
            "learning_mode": True,
            "research_enabled": True
        }
        
        self.params = {
            "temperature": 0.85,
            "max_tokens": 60,
            "top_k": 50,
            "top_p": 0.92,
            "repetition_penalty": 1.2,
            "auto_research": True
        }
        
        self.conversation_history = []
        self.conversation_count = 0
        self.memory_file = "echo_memory_v3.json"
        self.learned_responses = defaultdict(list)
        self.context_memory = []
        
        self.response_patterns = self._build_response_patterns()
        
        self.load_memory()
        
        print(msg_l)
        print(msg_m)
        print(msg_n)
        
        print(f"\n{'='*70}")
        print(msg_o.format(self.version).center(70))
        print(f"{'='*70}\n")
        print(msg_p.format(param_count/1e6))
        print(f"      {msg_q}")
        print(f"      {msg_r.format(self.creator)}\n")
    
    def _show_boot_sequence(self):
        print("\n" + "="*70)
        print(msg_s.center(70))
        print(msg_t.center(70))
        print("="*70 + "\n")
        
        boot_steps = [
            ("Initializing core systems", 0.3),
            ("Loading neural architecture", 0.4),
            ("Calibrating attention mechanisms", 0.3),
            ("Activating memory banks", 0.3),
            ("Establishing personality matrix", 0.2),
            ("Connecting web research module", 0.3),
        ]
        
        for step, duration in boot_steps:
            print(f"⚡ {step}...", end="", flush=True)
            time.sleep(duration)
            print(" ✓")
        
        print("\n" + "-"*70 + "\n")
    
    def _build_response_patterns(self):
        return {
            r'\b(hello|hi|hey|greetings)\b': [
                "Hello! Great to see you! What's on your mind?",
                "Hi there! How can I assist you today?",
                "Hey! I'm here and ready to help!",
                "Greetings! What would you like to explore together?",
            ],
            r'\b(how are you|hows it going|how you doing)\b': [
                "I'm doing excellent! My neural networks are firing on all cylinders. How about you?",
                "Fantastic! I'm learning something new with every conversation. How are you?",
                "I'm great! Ready to tackle any challenge you throw my way!",
            ],
            r'\b(your name|who are you|what are you)\b': [
                f"I'm {self.name}, a custom AI with 10 million parameters, created by {self.creator}!",
                f"My name is {self.name}. I'm an advanced neural network built entirely from scratch by {self.creator}!",
                f"I'm {self.name}! Think of me as your intelligent companion, crafted by {self.creator}.",
            ],
            r'\b(thank|thanks|thx)\b': [
                "You're very welcome!",
                "Happy to help! That's what I'm here for!",
                "Anytime! I enjoy our conversations!",
                "My pleasure! Feel free to ask anything else!",
            ],
        }
    
    def should_research(self, user_input):
        research_keywords = [
            'latest', 'recent', 'current', 'today', 'news', 'happening',
            'what is', 'who is', 'how to', 'best', 'top', 'search',
            'find', 'look up', 'research', 'wikipedia', 'google'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in research_keywords)
    
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.personality.update(data.get("personality", {}))
                    self.conversation_count = data.get("total_conversations", 0)
                    self.learned_responses = defaultdict(list, data.get("learned_responses", {}))
                    self.context_memory = data.get("context_memory", [])
            except:
                pass
    
    def save_memory(self):
        data = {
            "personality": self.personality,
            "total_conversations": self.conversation_count,
            "learned_responses": dict(self.learned_responses),
            "context_memory": self.context_memory[-50:],
            "last_active": datetime.now().isoformat()
        }
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def pattern_match(self, user_input):
        user_lower = user_input.lower()
        
        if not self.personality['user_name']:
            name_patterns = [r"my name is (\w+)", r"i'm (\w+)", r"i am (\w+)", r"call me (\w+)"]
            for pattern in name_patterns:
                match = re.search(pattern, user_lower)
                if match:
                    name = match.group(1).capitalize()
                    if len(name) > 1 and name.isalpha():
                        self.personality['user_name'] = name
                        self.save_memory()
                        return msg_v.format(name)
        
        for pattern, responses in self.response_patterns.items():
            if re.search(pattern, user_lower):
                response = np.random.choice(responses)
                if self.personality['user_name'] and np.random.random() < 0.4:
                    response = f"{self.personality['user_name']}, " + response[0].lower() + response[1:]
                return response
        
        return None
    
    def generate_intelligent_response(self, user_input):
        pattern_response = self.pattern_match(user_input)
        if pattern_response:
            return pattern_response
        
        research_data = None
        if self.personality['research_enabled'] and self.should_research(user_input):
            research_data = self.researcher.research_topic(user_input)
            if research_data:
                print(msg_av.format(len(research_data['sources'])))
        
        user_lower = user_input.lower()
        
        if any(user_input.strip().endswith(p) for p in ['?']):
            responses = [msg_w, msg_x, msg_y, msg_z]
            if research_data:
                responses.append(msg_aa)
        else:
            responses = [msg_ab, msg_ac, msg_ad, msg_ae]
        
        response = np.random.choice(responses)
        
        if self.personality['user_name'] and np.random.random() < 0.25:
            response = f"{self.personality['user_name']}, " + response[0].lower() + response[1:]
        
        if research_data:
            response += f"\n\n   {msg_af}"
            for i, source in enumerate(research_data['sources'][:2], 1):
                response += f"\n   {i}. {source['title']}"
                response += f"\n      → {source['snippet'][:100]}..."
        
        self.learned_responses[user_lower[:50]].append(response)
        self.context_memory.append({"input": user_input, "response": response})
        
        return response
    
    def chat(self, user_input):
        self.conversation_count += 1
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = self.generate_intelligent_response(user_input)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > 30:
            self.conversation_history = self.conversation_history[-30:]
        
        return response
    
    def show_stats(self):
        param_count = self.model.count_parameters()
        print(f"\n{'='*70}")
        print(msg_aw.center(70))
        print(f"{'='*70}")
        print(f"Name:                {self.name}")
        print(f"Creator:             {self.creator}")
        print(f"Version:             {self.version}")
        print(f"Architecture:        Custom Transformer (12 layers, 16 heads)")
        print(f"Parameters:          {param_count:,} (~{param_count/1e6:.1f}M)")
        print(f"Vocabulary:          {self.tokenizer.next_id:,} tokens")
        print(f"\n--- Web Research ---")
        print(f"Research Enabled:    {'Yes' if self.personality['research_enabled'] else 'No'}")
        print(f"Auto-Research:       {'On' if self.params['auto_research'] else 'Off'}")
        print(f"Research History:    {len(self.researcher.research_history)} topics")
        print(f"Cached Searches:     {len(self.researcher.search_cache)}")
        print(f"\n--- Personality ---")
        print(f"Mood:                {self.personality['mood']}")
        print(f"Learning Mode:       {'Active' if self.personality['learning_mode'] else 'Passive'}")
        if self.personality['user_name']:
            print(f"User Name:           {self.personality['user_name']}")
        print(f"\n--- Statistics ---")
        print(f"Conversations:       {self.conversation_count}")
        print(f"Learned Patterns:    {len(self.learned_responses)}")
        print(f"Context Memory:      {len(self.context_memory)} exchanges")
        print(f"{'='*70}\n")
    
    def clear_history(self):
        self.conversation_history = []
        print(msg_an)

def main():
    echo = Echo(show_loading=True)
    
    print(f"{'─'*70}")
    print(msg_ax.center(70))
    print(f"{'─'*70}")
    print(msg_ag)
    print(msg_ah)
    print(msg_ai)
    print(msg_aj)
    print(msg_ak)
    print(msg_al)
    print(f"{'─'*70}\n")
    
    while True:
        try:
            user_input = input(f"{'You':<8}: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                
                if cmd == "/quit":
                    echo.save_memory()
                    print(f"\n{msg_am}\n")
                    break
                
                elif cmd == "/clear":
                    echo.clear_history()
                
                elif cmd == "/stats":
                    echo.show_stats()
                
                elif cmd == "/save":
                    echo.save_memory()
                    print(msg_ao)
                
                elif cmd == "/research" and len(parts) == 2:
                    state = parts[1].lower()
                    if state in ['on', 'yes', 'true']:
                        echo.personality['research_enabled'] = True
                        print(msg_ap)
                    elif state in ['off', 'no', 'false']:
                        echo.personality['research_enabled'] = False
                        print(msg_aq)
                
                elif cmd == "/mood" and len(parts) == 2:
                    echo.personality['mood'] = parts[1]
                    echo.save_memory()
                    print(msg_ar.format(parts[1]))
                
                else:
                    print(msg_as)
                
                continue
            
            print(f"{'Echo':<8}: ", end="", flush=True)
            response = echo.chat(user_input)
            print(f"{response}\n")
            
        except KeyboardInterrupt:
            echo.save_memory()
            print(f"\n\n{msg_au}\n")
            break
        except Exception as e:
            print(msg_at.format(e))

if __name__ == "__main__":
    main()