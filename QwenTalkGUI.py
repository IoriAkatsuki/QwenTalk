import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import queue
import time
import os
import sys
from QwenTalk import ChatEngine

# --- High-DPI support for Windows ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except (ImportError, AttributeError):
    # This is only for Windows. On other OS, it will be ignored.
    pass

# Set customtkinter appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ChatGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QwenTalk GUI")
        self.geometry("1000x800")

        self.chat_engine = None
        self.response_queue = queue.Queue()
        self.base_font_size = 13
        self.font_scale = 1.0

        # --- Configure grid layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="QwenTalk", font=self.get_font(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Model & Device Controls
        self.model_path_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Path to Model")
        self.model_path_entry.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="ew")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path_entry.insert(0, os.path.join(script_dir, "Qwen3-8B-nf4-ov"))
        self.device_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["NPU", "GPU", "CPU"])
        self.device_menu.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load Model", command=self.load_model_thread)
        self.load_button.grid(row=3, column=0, padx=20, pady=(5, 20), sticky="ew")

        # Session Management
        self.session_label = ctk.CTkLabel(self.sidebar_frame, text="Conversations:", anchor="w")
        self.session_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.session_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Default"], command=self.switch_session)
        self.session_menu.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        self.new_session_button = ctk.CTkButton(self.sidebar_frame, text="New Chat", command=self.create_new_session)
        self.new_session_button.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        self.delete_session_button = ctk.CTkButton(self.sidebar_frame, text="Delete Current Chat", command=self.delete_current_session, fg_color="#D32F2F", hover_color="#B71C1C")
        self.delete_session_button.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        self.clear_button = ctk.CTkButton(self.sidebar_frame, text="Clear Current Chat", command=self.clear_history)
        self.clear_button.grid(row=8, column=0, padx=20, pady=5, sticky="ew")

        # --- Thinking Mode ---
        self.mode_label = ctk.CTkLabel(self.sidebar_frame, text="Response Modes:", anchor="w")
        self.mode_label.grid(row=9, column=0, padx=20, pady=(20, 0))

        self.creative_mode_switch = ctk.CTkSwitch(self.sidebar_frame, text="Creative Mode", command=self.toggle_creative_mode)
        self.creative_mode_switch.grid(row=10, column=0, padx=20, pady=10, sticky="w")
        self.creative_mode_switch.select() # Default to ON

        self.search_switch = ctk.CTkSwitch(self.sidebar_frame, text="Web Search", command=self.toggle_search_mode)
        self.search_switch.grid(row=11, column=0, padx=20, pady=10, sticky="w")

        self.no_think_switch = ctk.CTkSwitch(self.sidebar_frame, text="Direct Response", command=self.toggle_no_think_mode)
        self.no_think_switch.grid(row=12, column=0, padx=20, pady=10, sticky="w")


        # UI Scaling
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="Font Size:", anchor="w")
        self.scaling_label.grid(row=13, column=0, padx=20, pady=(10, 0))
        self.scaling_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.8, to=2.0, command=self.update_font_scaling)
        self.scaling_slider.set(1.0)
        self.scaling_slider.grid(row=14, column=0, padx=20, pady=(5, 20), sticky="ew")

        # --- Main Chat Area ---
        self.chat_frame = ctk.CTkFrame(self, corner_radius=10)
        self.chat_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.textbox = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state='disabled', borderwidth=0, selectbackground="#36719F")
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.tag_config('user', foreground='#007AFF')
        self.textbox.tag_config('bot', foreground='#34C759')
        self.textbox.tag_config('error', foreground='#FF3B30')
        self.textbox.tag_config('info', foreground='#8E8E93')

        # --- Input Frame ---
        self.entry = ctk.CTkEntry(self, placeholder_text="Type your message here...")
        self.entry.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="nsew")
        self.entry.bind("<Return>", self.send_message_event)
        
        # --- Status Bar ---
        self.status_bar = ctk.CTkFrame(self, height=25, corner_radius=0)
        self.status_bar.grid(row=3, column=1, padx=10, pady=(0,10), sticky="ew")
        self.status_label = ctk.CTkLabel(self.status_bar, text="Status: Please load a model to begin.", anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=10)
        self.perf_label = ctk.CTkLabel(self.status_bar, text="", anchor='e')
        self.perf_label.pack(side=tk.RIGHT, padx=10)

        self.update_font_scaling(1.0) # Apply initial font sizes
        self.set_initial_ui_state()
        self.after(100, self.check_queue)

    def get_font(self, size=None, weight="normal"):
        actual_size = int((size or self.base_font_size) * self.font_scale)
        return ctk.CTkFont(family="Microsoft YaHei", size=actual_size, weight=weight)

    def update_font_scaling(self, scale_value):
        self.font_scale = float(scale_value)
        # Update all fonts
        self.logo_label.configure(font=self.get_font(size=20, weight="bold"))
        self.model_path_entry.configure(font=self.get_font())
        self.device_menu.configure(font=self.get_font())
        self.load_button.configure(font=self.get_font())
        self.session_label.configure(font=self.get_font())
        self.session_menu.configure(font=self.get_font())
        self.new_session_button.configure(font=self.get_font())
        self.clear_button.configure(font=self.get_font())
        self.scaling_label.configure(font=self.get_font())
        self.entry.configure(font=self.get_font(size=14))
        self.status_label.configure(font=self.get_font(size=10))
        self.perf_label.configure(font=self.get_font(size=10))
        
        # ScrolledText needs special handling
        self.textbox.configure(font=("Microsoft YaHei", int(self.base_font_size * self.font_scale)))
        self.textbox.tag_config('bold', font=("Segoe UI", int(self.base_font_size * self.font_scale), "bold"))
        
        # Update textbox colors for theme changes
        bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        fg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        self.textbox.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)

    def set_initial_ui_state(self):
        self.entry.configure(state="disabled")
        self.session_menu.configure(state="disabled")
        self.new_session_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")

    def load_model_thread(self):
        self.load_button.configure(state="disabled", text="Loading...")
        self.status_label.configure(text="Status: Loading model, please wait...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            model_path = self.model_path_entry.get()
            device = self.device_menu.get()
            self.chat_engine = ChatEngine(model_path=model_path, device=device)
            self.chat_engine.load_model()
            self.response_queue.put(("load_success", None))
        except Exception as e:
            self.response_queue.put(("load_error", str(e)))

    def send_message_event(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input or not self.chat_engine:
            return

        self._update_textbox(f"You: \n", ('user', 'bold'))
        self._update_textbox(f"{user_input}\n\n", 'user')
        self.entry.delete(0, tk.END)

        self.entry.configure(state="disabled")
        self.status_label.configure(text="Status: Generating response...")
        self.perf_label.configure(text="")

        threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def generate_response(self, user_input):
        start_time = time.time()
        full_response = ""
        try:
            for chunk in self.chat_engine.generate_stream(user_input):
                self.response_queue.put(("stream_chunk", chunk))
                full_response += chunk
            end_time = time.time()
            token_count = self.chat_engine.count_tokens(full_response)
            self.response_queue.put(("stream_end", (token_count, time.time() - start_time)))
        except Exception as e:
            self.response_queue.put(("error", f"Error during generation: {e}"))

    def check_queue(self):
        try:
            while True:
                msg_type, content = self.response_queue.get_nowait()
                if msg_type == "load_success": self.handle_load_success()
                elif msg_type == "load_error": self.handle_load_error(content)
                elif msg_type == "stream_chunk": self.handle_stream_chunk(content)
                elif msg_type == "stream_end": self.handle_stream_end(content)
                elif msg_type == "error": self.handle_error(content)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_queue)

    def handle_load_success(self):
        self.status_label.configure(text=f"Status: Ready (Model loaded on {self.chat_engine.device})")
        self.load_button.configure(state="normal", text="Load Model")
        self.entry.configure(state="normal")
        self.session_menu.configure(state="normal")
        self.new_session_button.configure(state="normal")
        self.clear_button.configure(state="normal")
        self.update_session_menu()
        self.display_current_history()
        self._update_textbox("Model loaded. Start chatting or create a new conversation.\n", 'info')

    def handle_load_error(self, error_msg):
        self.status_label.configure(text="Status: Error loading model.")
        self._update_textbox(f"Error: {error_msg}\n", 'error')
        self.load_button.configure(state="normal", text="Load Model")

    def handle_stream_chunk(self, chunk):
        if not getattr(self, 'bot_tag_applied', False):
            self._update_textbox(f"Bot: \n", ('bot', 'bold'))
            self.bot_tag_applied = True
        self._update_textbox(chunk, 'bot')

    def handle_stream_end(self, stats):
        self.bot_tag_applied = False
        self._update_textbox("\n\n", 'bot')
        self.entry.configure(state="normal")
        self.status_label.configure(text="Status: Ready")
        token_count, gen_time = stats
        self.perf_label.configure(text=f"Tokens: {token_count} | Time: {gen_time:.2f}s")

    def handle_error(self, error_msg):
        self._update_textbox(f"ERROR: {error_msg}\n\n", 'error')
        self.entry.configure(state="normal")
        self.status_label.configure(text="Status: Error")

    def _update_textbox(self, text, tags=None):
        self.textbox.configure(state='normal')
        self.textbox.insert(tk.END, text, tags)
        self.textbox.configure(state='disabled')
        self.textbox.see(tk.END)

    def clear_history(self):
        if self.chat_engine:
            self.chat_engine.clear_history()
        self.display_current_history()
        self._update_textbox("Current chat history cleared.\n", 'info')

    def create_new_session(self):
        dialog = ctk.CTkInputDialog(text="Enter new conversation name:", title="New Chat")
        new_name = dialog.get_input()
        if new_name and self.chat_engine:
            success, message = self.chat_engine.create_session(new_name)
            if success:
                self.update_session_menu()
                self.display_current_history()
            else:
                # TODO: Show a proper error dialog
                print(f"Failed to create session: {message}")

    def delete_current_session(self):
        if not self.chat_engine:
            return
        
        session_to_delete = self.chat_engine.get_active_session_name()
        
        success, message = self.chat_engine.delete_session(session_to_delete)
        
        if success:
            self.update_session_menu()
            self.display_current_history()
            self._update_textbox(f"Chat '{session_to_delete}' deleted.\n", 'info')
        else:
            self._update_textbox(f"Error: {message}\n", 'error')

    def toggle_creative_mode(self):
        if self.chat_engine:
            is_creative = self.creative_mode_switch.get() == 1
            self.chat_engine.set_creative_mode(is_creative)
            mode = "Creative" if is_creative else "Deterministic"
            self.status_label.configure(text=f"Status: Switched to {mode} mode.")

    def toggle_search_mode(self):
        if not self.chat_engine:
            return
        is_enabled = self.search_switch.get() == 1
        print(f"[DEBUG] GUI: toggle_search_mode called. Switch is_enabled: {is_enabled}") # DEBUG
        self.chat_engine.set_search_mode(is_enabled)
        if is_enabled:
            self.no_think_switch.deselect()
            self.chat_engine.set_no_think_mode(False)
            self.status_label.configure(text="Status: Web Search mode enabled.")
        else:
            self.status_label.configure(text="Status: Web Search mode disabled.")

    def toggle_no_think_mode(self):
        if not self.chat_engine:
            return
        is_enabled = self.no_think_switch.get() == 1
        self.chat_engine.set_no_think_mode(is_enabled)
        if is_enabled:
            self.search_switch.deselect()
            self.chat_engine.set_search_mode(False)
            self.status_label.configure(text="Status: Direct Response mode enabled.")
        else:
            self.status_label.configure(text="Status: Direct Response mode disabled.")

    def switch_session(self, session_name):
        if self.chat_engine:
            self.chat_engine.switch_session(session_name)
            self.display_current_history()

    def update_session_menu(self):
        if self.chat_engine:
            sessions = self.chat_engine.list_sessions()
            active_session = self.chat_engine.get_active_session_name()
            self.session_menu.configure(values=sessions)
            self.session_menu.set(active_session)

    def display_current_history(self):
        self.textbox.configure(state='normal')
        self.textbox.delete('1.0', tk.END)
        if self.chat_engine:
            history = self.chat_engine.get_history()
            for msg in history:
                role = msg['role']
                name = "You" if role == "user" else "Bot"
                self._update_textbox(f"{name}: \n", (role, 'bold'))
                self._update_textbox(f"{msg['content']}\n\n", role)
        self.textbox.configure(state='disabled')

if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()