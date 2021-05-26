from __future__ import division
from json import load

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

import uuid
import json
import api.dialogFlowAPI as API # 안녕이 서버 모듈 임포트

# 실시간 스트림 서비스는 최대 5분 까지만 제공이 가능하다.

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """녹음 스트림을 생성한다."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses):
    
    isStarted = False
    conversations = []

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # 최종적으로 변환된 스크립트
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            # 안녕이 시작하기
            # 다음의 단어로 시작 가능
            # start keyword: 시작, 좋은 아침, 안녕, 나왔어
            # 한번만 작동시키기 위해서 (isStarted == False) 조건 추가
            if re.search(r"\b(시작|좋은 아침|안녕|나왔어)\b", transcript, re.I) and isStarted == False:
                isStarted = True
                print('안녕이 서비스를 시작합니다.')
                print('==================================')
                continue

            # 안녕이 서비스 동작 중인 경우
            if isStarted:
                # 안녕이 종료하기
                # 다음의 단어로 종료 가능
                # stop keyword: 종료, 잘자, 잘게, 잔다, 바이바이
                if re.search(r"\b(종료|잘자|잘게|잔다|바이바이)\b", transcript, re.I):
                    print("안녕이 서비스를 종료합니다.")
                    # 대화 저장하기
                    with open('conversation.json', 'w') as f:
                        f.write(json.dumps(conversations, ensure_ascii=False))

                    print("총 " + str(len(conversations)) + "건의 대화가 저장되었습니다.")
                    isStarted = False
                    continue

                else:
                    print('사용자: ' + transcript + overwrite_chars)
                    
                    # 안녕이에게 답장 받아오기
                    answer = API.getAnswer(transcript)
                    res_data = API.getResponseData(transcript)
                    print('안녕이: ' + answer)
                    print('감정: ' + API.getEmotion(res_data))
                    print('----------------')
                    
                    # 대화 저장
                    UUID = str(uuid.uuid1()) # 객체 구별을 위한 랜덤 값 생성
                    conversation = { "id": UUID,"user": transcript, "answer": answer, "emotion": API.getEmotion(res_data)}
                    conversations.append(conversation)

            
            num_chars_printed = 0

def main():
    print('============= 안녕이 v1 =============')

    # 언어 설정 = ko(한국어)
    # 다른 나라 언어는 아래 링크 참고(들어가서 BCP-47 태그 확인)
    # http://g.co/cloud/speech/docs/languages
    language_code = "ko"

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    # 마이크 스트림 생성
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # 실제로 화면에 뿌려주는 부분
        listen_print_loop(responses)

# python -m [파일명] 으로 실행할 수 있도록 설정
# 배포시 따로 path 설정을 하지 않아도 알아서 name 으로 찾아준다.
if __name__ == "__main__":
    main()