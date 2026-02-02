"""
Visualization GUI for RL Game AI.
Displays screenshot, pre-processed screen, game states, and model actions.
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Optional, Dict, Any


class VisualizationGUI:
    """
    GUI for visualizing the RL agent's perception and decisions.

    Layout:
    +------------------+------------------+
    |   Screenshot     | Pre-processed    |
    |   (Raw Input)    |    Screen        |
    +------------------+------------------+
    |   Game States    | Action Choice    | Placeholder |
    +------------------+------------------+-------------+
    """

    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize the visualization GUI.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self.window = tk.Tk()
        self.window.title("RL Game AI - Visualization")
        self.window.geometry(f"{width}x{height}")
        self.window.configure(bg='#2b2b2b')

        # Image references (prevent garbage collection)
        self._screenshot_photo: Optional[ImageTk.PhotoImage] = None
        self._processed_photo: Optional[ImageTk.PhotoImage] = None

        # Display sizes
        self.display_width = 400
        self.display_height = 300

        self._setup_ui()

    def _setup_ui(self):
        """Setup all UI components."""
        # Configure grid: 3 columns for bottom row, top row uses frames
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.columnconfigure(2, weight=1)
        self.window.rowconfigure(0, weight=2)  # Top row gets more space
        self.window.rowconfigure(1, weight=1)  # Bottom row

        # Create a top frame that spans all columns for 50/50 split
        self.top_frame = ttk.Frame(self.window)
        self.top_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=0, pady=0)
        self.top_frame.columnconfigure(0, weight=1)  # Left 50%
        self.top_frame.columnconfigure(1, weight=1)  # Right 50%
        self.top_frame.rowconfigure(0, weight=1)

        # Style configuration
        style = ttk.Style()
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white')
        style.configure('Dark.TLabelframe', background='#2b2b2b', foreground='white')
        style.configure('Dark.TLabelframe.Label', background='#2b2b2b', foreground='#00ff00')

        # ================================================================
        # TOP-LEFT: Screenshot Display
        # ================================================================
        self._setup_screenshot_panel()

        # ================================================================
        # TOP-RIGHT: Pre-processed Screen Display
        # ================================================================
        self._setup_processed_panel()

        # ================================================================
        # BOTTOM-LEFT: Game States Display
        # ================================================================
        self._setup_game_states_panel()

        # ================================================================
        # BOTTOM-CENTER: Action Choice Display
        # ================================================================
        self._setup_action_panel()

        # ================================================================
        # BOTTOM-RIGHT: Placeholder
        # ================================================================
        self._setup_placeholder_panel()

    def _setup_screenshot_panel(self):
        """Setup the raw screenshot display panel."""
        frame = ttk.LabelFrame(
            self.top_frame,
            text=" 📷 Raw Screenshot ",
            padding="5"
        )
        frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Canvas for screenshot
        self.screenshot_canvas = tk.Canvas(
            frame,
            width=self.display_width,
            height=self.display_height,
            bg='#1a1a1a',
            highlightthickness=1,
            highlightbackground='#444444'
        )
        self.screenshot_canvas.grid(row=0, column=0, sticky='nsew')

        # Placeholder text
        self.screenshot_canvas.create_text(
            self.display_width // 2,
            self.display_height // 2,
            text="No screenshot available",
            fill='#666666',
            font=('Consolas', 12),
            tags='placeholder'
        )

    def _setup_processed_panel(self):
        """Setup the pre-processed screen display panel."""
        frame = ttk.LabelFrame(
            self.top_frame,
            text=" 🔧 Pre-processed Screen ",
            padding="5"
        )
        frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Canvas for processed image
        self.processed_canvas = tk.Canvas(
            frame,
            width=self.display_width,
            height=self.display_height,
            bg='#1a1a1a',
            highlightthickness=1,
            highlightbackground='#444444'
        )
        self.processed_canvas.grid(row=0, column=0, sticky='nsew')

        # Placeholder text
        self.processed_canvas.create_text(
            self.display_width // 2,
            self.display_height // 2,
            text="No processed image available",
            fill='#666666',
            font=('Consolas', 12),
            tags='placeholder'
        )

    def _setup_game_states_panel(self):
        """Setup the game states information panel."""
        frame = ttk.LabelFrame(
            self.window,
            text=" 📊 Game States ",
            padding="10"
        )
        frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Text widget for game states
        self.states_text = tk.Text(
            frame,
            width=35,
            height=12,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Consolas', 10),
            insertbackground='white',
            highlightthickness=1,
            highlightbackground='#444444',
            state='disabled'
        )
        self.states_text.grid(row=0, column=0, sticky='nsew')

        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.states_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.states_text.configure(yscrollcommand=scrollbar.set)

        # Initial placeholder text
        self._update_text_widget(self.states_text, "Waiting for game state data...")

    def _setup_action_panel(self):
        """Setup the action choice display panel."""
        frame = ttk.LabelFrame(
            self.window,
            text=" 🎮 Model Action ",
            padding="10"
        )
        frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)

        # Large action display
        self.action_label = tk.Label(
            frame,
            text="---",
            font=('Arial', 32, 'bold'),
            bg='#1a1a1a',
            fg='#ffcc00',
            width=12,
            height=3
        )
        self.action_label.grid(row=0, column=0, sticky='nsew', pady=10)

        # Action details / confidence
        self.action_details = tk.Label(
            frame,
            text="Confidence: ---%\nAction ID: ---",
            font=('Consolas', 10),
            bg='#2b2b2b',
            fg='#aaaaaa',
            justify='center'
        )
        self.action_details.grid(row=1, column=0, sticky='ew')

    def _setup_placeholder_panel(self):
        """Setup the placeholder panel for future use."""
        frame = ttk.LabelFrame(
            self.window,
            text=" 📦 Reserved ",
            padding="10"
        )
        frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Placeholder label
        placeholder_label = tk.Label(
            frame,
            text="Future Feature\n\n(Placeholder)",
            font=('Arial', 12),
            bg='#1a1a1a',
            fg='#555555',
            justify='center'
        )
        placeholder_label.grid(row=0, column=0, sticky='nsew')

    def _update_text_widget(self, widget: tk.Text, text: str):
        """Helper to update a disabled text widget."""
        widget.configure(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    # ====================================================================
    # PUBLIC UPDATE METHODS
    # ====================================================================

    def update_screenshot(self, image: np.ndarray):
        """
        Update the screenshot display.

        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        # Clear placeholder
        self.screenshot_canvas.delete('placeholder')

        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Get current canvas size
        canvas_width = self.screenshot_canvas.winfo_width()
        canvas_height = self.screenshot_canvas.winfo_height()

        # Use default if canvas not yet rendered
        if canvas_width <= 1:
            canvas_width = self.display_width
        if canvas_height <= 1:
            canvas_height = self.display_height

        # Calculate aspect-ratio-preserving resize (fit to canvas)
        img_width, img_height = pil_image.size
        img_aspect = img_width / img_height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            # Image is wider than canvas - fit to width
            new_width = canvas_width
            new_height = int(canvas_width / img_aspect)
        else:
            # Image is taller than canvas - fit to height
            new_height = canvas_height
            new_width = int(canvas_height * img_aspect)

        # Resize maintaining aspect ratio
        pil_image = pil_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Convert to PhotoImage
        self._screenshot_photo = ImageTk.PhotoImage(pil_image)

        # Update canvas - center the image
        self.screenshot_canvas.delete('all')
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.screenshot_canvas.create_image(
            x_offset, y_offset,
            anchor='nw',
            image=self._screenshot_photo
        )

    def update_processed_screen(self, image: np.ndarray):
        """
        Update the pre-processed screen display.

        Args:
            image: Grayscale or RGB image as numpy array
                   Can be (H, W), (H, W, 1), or (H, W, 3)
        """
        # Clear placeholder
        self.processed_canvas.delete('placeholder')

        # Handle different input formats
        if len(image.shape) == 2:
            # Grayscale (H, W)
            pass
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            # Grayscale with channel dim (H, W, 1)
            image = image.squeeze(-1)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            # RGB - convert to grayscale for display consistency
            pass

        # Normalize to 0-255
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to PIL Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        else:
            pil_image = Image.fromarray(image)

        # Get current canvas size
        canvas_width = self.processed_canvas.winfo_width()
        canvas_height = self.processed_canvas.winfo_height()

        # Use default if canvas not yet rendered
        if canvas_width <= 1:
            canvas_width = self.display_width
        if canvas_height <= 1:
            canvas_height = self.display_height

        # Calculate aspect-ratio-preserving resize (fit to canvas)
        img_width, img_height = pil_image.size
        img_aspect = img_width / img_height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            # Image is wider - fit to width
            new_width = canvas_width
            new_height = int(canvas_width / img_aspect)
        else:
            # Image is taller or square - fit to height
            new_height = canvas_height
            new_width = int(canvas_height * img_aspect)

        # Resize maintaining aspect ratio
        pil_image = pil_image.resize(
            (new_width, new_height),
            Image.Resampling.NEAREST  # Use NEAREST for pixel-perfect view
        )

        # Convert grayscale to RGB for display
        if pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')

        # Convert to PhotoImage
        self._processed_photo = ImageTk.PhotoImage(pil_image)

        # Update canvas - center the image
        self.processed_canvas.delete('all')
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.processed_canvas.create_image(
            x_offset, y_offset,
            anchor='nw',
            image=self._processed_photo
        )

    def update_game_states(self, states: Dict[str, Any]):
        """
        Update the game states display.

        Args:
            states: Dictionary of game state key-value pairs
        """
        # Format states as readable text
        lines = ["=" * 30]
        lines.append("  GAME STATE INFO")
        lines.append("=" * 30)
        lines.append("")

        for key, value in states.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            elif isinstance(value, bool):
                status = "✓ Yes" if value else "✗ No"
                lines.append(f"  {key}: {status}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("=" * 30)

        text = "\n".join(lines)
        self._update_text_widget(self.states_text, text)

    def update_action(self, action_name: str, action_id: int = -1, confidence: float = -1.0):
        """
        Update the action choice display.

        Args:
            action_name: Human-readable action name (e.g., "JUMP", "LEFT", "ATTACK")
            action_id: Numeric action ID
            confidence: Model confidence (0-1), or -1 if not available
        """
        # Update main action label
        self.action_label.configure(text=action_name.upper())

        # Update details
        if confidence >= 0:
            conf_text = f"Confidence: {confidence * 100:.1f}%"
        else:
            conf_text = "Confidence: N/A"

        if action_id >= 0:
            id_text = f"Action ID: {action_id}"
        else:
            id_text = "Action ID: N/A"

        self.action_details.configure(text=f"{conf_text}\n{id_text}")

        # Color-code based on confidence
        if confidence >= 0.8:
            self.action_label.configure(fg='#00ff00')  # Green - high confidence
        elif confidence >= 0.5:
            self.action_label.configure(fg='#ffcc00')  # Yellow - medium
        elif confidence >= 0:
            self.action_label.configure(fg='#ff6600')  # Orange - low
        else:
            self.action_label.configure(fg='#ffcc00')  # Default yellow

    def update(self):
        """Process pending GUI events (call this in your main loop)."""
        self.window.update()

    def run(self):
        """Start the GUI main loop (blocking)."""
        self.window.mainloop()

    def close(self):
        """Close the GUI window."""
        self.window.destroy()


# ============================================================================
# DEMO / TEST
# ============================================================================

def demo():
    """Demonstrate the visualization GUI with sample data."""
    # import time

    # Create GUI
    gui = VisualizationGUI()

    # Create sample data
    # Fake screenshot (random colored noise)
    fake_screenshot = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Fake processed image (84x84 grayscale)
    fake_processed = np.random.rand(84, 84).astype(np.float32)

    # Fake game states
    fake_states = {
        'score': 1250,
        'health': 75.5,
        'max_health': 100,
        'ammo': 24,
        'position_x': 123.45,
        'position_y': 678.90,
        'game_over': False,
        'level': 3,
        'enemies_killed': 15,
    }

    # Actions to cycle through
    actions = ['IDLE', 'MOVE_LEFT', 'MOVE_RIGHT', 'JUMP', 'ATTACK', 'DEFEND']
    action_idx = 0

    # Update GUI with sample data
    gui.update_screenshot(fake_screenshot)
    gui.update_processed_screen(fake_processed)
    gui.update_game_states(fake_states)
    gui.update_action(actions[0], action_id=0, confidence=0.85)

    def update_demo():
        """Periodic update for demo."""
        nonlocal action_idx, fake_states

        # Cycle through actions
        action_idx = (action_idx + 1) % len(actions)
        confidence = np.random.uniform(0.4, 0.99)
        gui.update_action(actions[action_idx], action_id=action_idx, confidence=confidence)

        # Update some game states
        fake_states['score'] += np.random.randint(0, 50)
        fake_states['health'] = max(0, fake_states['health'] - np.random.uniform(0, 2))
        fake_states['position_x'] += np.random.uniform(-5, 5)
        gui.update_game_states(fake_states)

        # Generate new random images
        gui.update_screenshot(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        gui.update_processed_screen(np.random.rand(84, 84).astype(np.float32))

        # Schedule next update
        gui.window.after(500, update_demo)

    # Start periodic updates
    gui.window.after(1000, update_demo)

    # Run GUI
    gui.run()


if __name__ == '__main__':
    demo()
