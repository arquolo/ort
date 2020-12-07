__all__ = ['Sound']

import time
from contextlib import ExitStack

import numpy as np
import wrapt

from .. import coroutine


@coroutine
def _iter_chunks(arr):
    i = 0
    chunk_size = yield
    while i < len(arr):
        i, chunk = i + chunk_size, arr[i: i + chunk_size]
        chunk_size = yield chunk


class Sound(wrapt.ObjectProxy):
    """Wraps numpy.array to be playable as sound

    Parameters:
    - rate - sample rate to use for playback

    Usage:
    ```
    from glow.io import Sound

    sound = Sound.load('test.flac')  # Array-like wrapper
    rate: int = sound.rate
    shape: Tuple[int, int] = sound.shape
    dtype: numpy.dtype = sound.dtype
    sound.play()  # Plays sound through default device
    ```
    """

    def __init__(self, array: np.ndarray, rate: int = 44100):
        assert array.ndim == 2
        assert array.shape[-1] in (1, 2)
        assert array.dtype in ('int8', 'int16', 'int32', 'float32')

        super().__init__(array)
        self._self_rate = rate

    @property
    def rate(self):
        return self._self_rate

    def __repr__(self) -> str:
        prefix = f'{type(self).__name__}('
        pad = ' ' * len(prefix)
        body = '\n'.join(
            f'{pad if i else prefix}{line}'
            for i, line in enumerate(f'{self.__wrapped__}'.splitlines()))
        return f'{body}, dtype={self.dtype}, rate={self.rate})'

    def play(self, chunk_size=1024) -> None:
        """Play array as sound"""
        import pyaudio
        pyaudio_type = f'pa{str(self.dtype).title()}'
        it = _iter_chunks(self)

        def callback(_, num_frames, _1, _2):
            try:
                return it.send(num_frames), pyaudio.paContinue
            except StopIteration:
                return None, pyaudio.paComplete

        with ExitStack() as stack:
            audio = pyaudio.PyAudio()
            stack.callback(audio.terminate)

            stream = audio.open(
                rate=self.rate,
                channels=self.shape[1],
                format=getattr(pyaudio, pyaudio_type),
                output=True,
                frames_per_buffer=chunk_size,
                stream_callback=callback)

            stack.callback(stream.close)
            stack.callback(stream.stop_stream)

            while stream.is_active():
                time.sleep(0.1)

    @classmethod
    def load(cls, path: str) -> 'Sound':
        assert str(path).endswith('.flac')
        import soundfile

        array, samplerate = soundfile.read(str(path))
        return cls(array.astype('float32'), samplerate)
