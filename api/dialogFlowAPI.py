from os import linesep
import requests
import json
import subprocess

"""
Dialogflow Annyeong-i API v1
"""

# @description: Access Token 받아오는 메서드
# @return: token (string)
def getAccessToken():
    command = '~/project/annyeong-i/gcloudGetAccessToken.sh'
    token = str(subprocess.check_output(
        command, encoding='utf-8', shell=True)).rstrip('\n')
    return token

# @description: 응답 전체 데이터 가져오는 메서드
# @param: input (string)
# @return: response_data (dict)
def getResponseData(input):
    _url = 'https://dialogflow.googleapis.com/v2/projects/annyeong-ydxv/agent/sessions/123456789:detectIntent'
    headers = {'Authorization': 'Bearer ' + getAccessToken(),
            'Content-Type': 'application/json; charset=utf-8'}
    data = {"query_input": {"text": {"text": input, "language_code": "ko"}}}
    
    # POST 요청 전송
    res = requests.post(_url, headers=headers, data=json.dumps(data))
    # 응답 데이터 받아서 dict 형태로 매핑시켜주기
    res_data = res.json()

    return res_data

# @description: 응답 데이터 중에서 답변만 가져오는 메서드
# @param: input (string)
# @return: output (stirng)
def getAnswer(input):
    res_data = getResponseData(input)
    output = res_data['queryResult']['fulfillmentMessages'][0]['text']['text'][0]
    return output

# @description: 응답 데이터 중에서 감정만 가져오는 메서드
# @param: data (dict)
# @return: emotion (stirng)
def getEmotion(data):
    emotion = data['queryResult']['intent']['displayName']
    return emotion

# @description: 부정적인 감정인지 판별하는 메서드
# @param: emotion (string)
# @return: True/False (boolean)
def isNagative(emotion):
    """
    TODO
      case1) 모든 감정을 '긍정', '부정' 으로 나눌지
      case2) (화남, 우울, 슬픔, 무기력 ..) => '부정'으로 할 것인지.
    """
    if(emotion == '부정표현'):
        # 부정적인 감정이면 True 반환
        return True
    else:
        # 긍정적인 감정이면 False 반환
        return False
 
# @description: 우울증 판정 메서드
# @return: True/False (boolean)
def isDepress():
    # TODO
    pass

#======================= 테스트 출력 =======================#
# print('=======================================')
# input = '밥 맛이 없네..'

# res_data = getResponseData(input)

# print(input)
# print(getAnswer(input))
# print(getEmotion(res_data))
# print(isNagative(getEmotion(res_data)))
