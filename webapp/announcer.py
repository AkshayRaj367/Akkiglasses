"""
==========================================================================
  announcer.py  –  Text-to-Speech Object Announcer for Smart Glasses
==========================================================================
Runs a background thread that speaks detected object names through the
system speaker / smart-glasses audio output.

Features:
  • Cooldown per object (avoids repeating "person, person, person…")
  • Configurable speech rate, volume, and voice
  • Priority announcements for new objects entering the scene
  • Thread-safe queue with deduplication
  • Graceful on/off toggle without restarting the stream

Author : Akshay Raj
Project: Smart Glasses – Object Detection
==========================================================================
"""

import threading
import time
import logging
import queue
from collections import defaultdict

logger = logging.getLogger("announcer")

# ── optional TTS import ──────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False
    logger.warning("pyttsx3 not installed – TTS disabled.  pip install pyttsx3")


class ObjectAnnouncer:
    """
    Background text-to-speech announcer for detected objects.

    Usage:
        announcer = ObjectAnnouncer()
        announcer.start()
        announcer.announce(detections)   # call from process loop
        announcer.stop()
    """

    # Default cooldown: don't repeat same label for N seconds
    DEFAULT_COOLDOWN = 5.0

    # Minimum confidence to announce
    MIN_CONFIDENCE = 0.45

    # Max objects to announce per frame batch
    MAX_PER_BATCH = 3

    def __init__(self, rate=175, volume=0.9, voice_index=0,
                 cooldown=None, enabled=True):
        """
        Parameters
        ----------
        rate : int
            Speech rate (words per minute).  Default 175.
        volume : float
            Volume 0.0 – 1.0.
        voice_index : int
            Index into system voice list (0 = David, 1 = Zira on Win).
        cooldown : float
            Seconds before re-announcing the same label.
        enabled : bool
            Whether announcements are active at startup.
        """
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._cooldown = cooldown or self.DEFAULT_COOLDOWN
        self._enabled = enabled

        self._engine = None
        self._queue = queue.Queue(maxsize=50)
        self._last_announced = defaultdict(float)  # label → timestamp
        self._known_labels = set()      # labels currently in scene
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        """Start the TTS background thread."""
        if not TTS_OK:
            logger.warning("TTS not available – announcer disabled")
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("ObjectAnnouncer started")

    def stop(self):
        """Stop the TTS background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None
        logger.info("ObjectAnnouncer stopped")

    # ── public API ────────────────────────────────────────────────────

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
        if not val:
            # Clear queue when disabled
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    @property
    def cooldown(self):
        return self._cooldown

    @cooldown.setter
    def cooldown(self, val: float):
        self._cooldown = max(1.0, val)

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, val: int):
        self._rate = val
        if self._engine:
            try:
                self._engine.setProperty("rate", val)
            except Exception:
                pass

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, val: float):
        self._volume = max(0.0, min(1.0, val))
        if self._engine:
            try:
                self._engine.setProperty("volume", self._volume)
            except Exception:
                pass

    def get_settings(self) -> dict:
        """Return current announcer settings."""
        return {
            "enabled": self._enabled,
            "cooldown": self._cooldown,
            "rate": self._rate,
            "volume": self._volume,
            "tts_available": TTS_OK,
        }

    def announce(self, detections: list[dict]):
        """
        Process a list of detection dicts and queue new objects for speech.

        Call this from the video processing loop on detection frames.
        Each detection dict should have 'label' and 'confidence' keys.
        """
        if not self._enabled or not TTS_OK:
            return

        now = time.time()
        current_labels = set()
        to_announce = []

        for det in detections:
            label = det.get("label", "").strip()
            conf  = det.get("confidence", 0)

            if not label or conf < self.MIN_CONFIDENCE:
                continue
            # Skip internal prefixes for cleaner speech
            clean = label.replace("[Custom] ", "").replace("_", " ")
            current_labels.add(clean)

            # Check cooldown
            last = self._last_announced.get(clean, 0)
            if (now - last) < self._cooldown:
                continue

            to_announce.append((clean, conf))

        # Prioritise by confidence, cap per batch
        to_announce.sort(key=lambda x: -x[1])
        for lbl, conf in to_announce[:self.MAX_PER_BATCH]:
            try:
                self._queue.put_nowait(lbl)
                self._last_announced[lbl] = now
            except queue.Full:
                break

        # Track scene changes
        with self._lock:
            self._known_labels = current_labels

    # ── background TTS loop ──────────────────────────────────────────

    def _make_engine(self):
        """Create and configure a fresh pyttsx3 engine."""
        eng = pyttsx3.init()
        eng.setProperty("rate", self._rate)
        eng.setProperty("volume", self._volume)
        voices = eng.getProperty("voices")
        if voices and self._voice_index < len(voices):
            eng.setProperty("voice", voices[self._voice_index].id)
        return eng

    def _run_loop(self):
        """Background thread: speaks queued labels.

        A fresh pyttsx3 engine is created for **every** utterance to work
        around a Windows SAPI5 bug where ``runAndWait()`` silently stops
        producing audio after the first call.
        """
        # Quick sanity check that the engine can be created at all
        try:
            test = self._make_engine()
            test.stop()
            del test
            logger.info(f"TTS engine ready (rate={self._rate}, "
                        f"vol={self._volume})")
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            return

        while not self._stop_event.is_set():
            try:
                label = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self._enabled:
                continue

            try:
                logger.info(f"Announcing: {label}")
                engine = self._make_engine()
                engine.say(label)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                logger.warning(f"TTS speak failed: {e}")

        # cleanup
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass


# ── singleton ────────────────────────────────────────────────────────────
announcer = ObjectAnnouncer()
