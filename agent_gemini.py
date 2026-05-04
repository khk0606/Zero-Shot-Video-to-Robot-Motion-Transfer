import requests
import json
import os
from pathlib import Path


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return api_key


def _default_prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"

class Agent():
    def __init__(self, system_prompt_path, cfg):
        system_prompt_path = str(system_prompt_path)

        # 1. 프롬프트 파일 로드
        if not os.path.exists(system_prompt_path):
            # 경로가 안 맞을 경우 prompts 폴더 안에서 찾아보기
            base_name = os.path.basename(system_prompt_path)
            alt_path = (_default_prompt_dir() / base_name).as_posix()
            if os.path.exists(alt_path):
                system_prompt_path = alt_path
            else:
                print(f"[Warning] Prompt file not found: {system_prompt_path}")
                self.system_prompt = "You are a helpful assistant."
        
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()

        self.cfg = cfg
        # 모델명 설정 (기본값을 리스트에 있는 gemini-2.0-flash로 변경)
        self.model_name = cfg.get('model', 'gemini-2.0-flash')
        
        # 2. 대화 기록 초기화 (System Prompt 포함)
        self.history = [
            {"role": "user", "parts": [{"text": self.system_prompt}]},
            {"role": "model", "parts": [{"text": "Understood. I am ready to act as the specified agent."}]}
        ]
        print(f"[Activated Agent] {self.__class__.__name__} (Model: {self.model_name})")

    def prepare_user_content(self, contents:list):
        parts = []
        for item in contents:
            if item["type"] == "text":
                parts.append({"text": item["data"]})
            elif item["type"] in ["image_uri", "image_url"]:
                # Base64 이미지를 inlineData로 변환
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": item["data"]
                    }
                })
        self.history.append({"role": "user", "parts": parts})

    def query(self):
        # 3. REST API URL 직접 호출 (v1beta)
        api_key = _get_api_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096
            }
        }

        try:
            # requests로 직접 전송 (라이브러리 버전 문제 해결)
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return f"Error: {response.text}"

            result = response.json()
            
            # 응답 텍스트 추출
            try:
                content_text = result['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                print(f"Unexpected response structure: {result}")
                content_text = "Error parsing response."

            # 대화 기록에 추가
            self.history.append({"role": "model", "parts": [{"text": content_text}]})
            return content_text

        except Exception as e:
            print(f"Exception during query: {e}")
            return "Error in generating response."

# --- 하위 클래스 (기존 유지) ---

class TaskDescriptor(Agent):
    def __init__(self, cfg, prompt_dir=None):
        prompt_dir = Path(prompt_dir) if prompt_dir else _default_prompt_dir()
        super().__init__(prompt_dir / "task_descriptor_system.txt", cfg)
    
    def analyse(self, encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri", "data": encoded_frame_grid}])
        return self.query()

class ContactSequenceAnalyser(Agent):
    def __init__(self, cfg, prompt_dir=None):
        prompt_dir = Path(prompt_dir) if prompt_dir else _default_prompt_dir()
        super().__init__(prompt_dir / "contact_sequence_system.txt", cfg)
        
    def analyse(self, encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri", "data": encoded_frame_grid}])
        self.query() # 1차 분석
        self.prepare_user_content([{
            "type":"text", 
            "data":"Revise the contact sequence you just generated. Check frame by frame to make sure it is correct."
        }])
        return self.query() # 2차 검증

class TaskRequirementAnalyser(Agent):
    def __init__(self, cfg, prompt_dir=None):
        prompt_dir = Path(prompt_dir) if prompt_dir else _default_prompt_dir()
        super().__init__(prompt_dir / "task_requirement_system.txt", cfg)
        
    def analyse(self, encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri", "data": encoded_frame_grid}])
        return self.query()

class GaitAnalyser(Agent):
    def __init__(self, cfg, prompt_dir=None):
        prompt_dir = Path(prompt_dir) if prompt_dir else _default_prompt_dir()
        super().__init__(prompt_dir / "gait_pattern_system.txt", cfg)
        
    def analyse(self, encoded_frame_grid, contact_pattern):
        self.prepare_user_content([
            {"type":"image_uri", "data": encoded_frame_grid},
            {"type":"text", "data": f"Likely contact pattern: {contact_pattern}"}
        ])
        return self.query()

class SUSGenerator(Agent):
    def __init__(self, cfg, prompt_dir=None):
        prompt_dir = Path(prompt_dir) if prompt_dir else _default_prompt_dir()
        super().__init__(prompt_dir / "SUS_generation_prompt.txt", cfg)
        self.cfg = cfg
        self.prompt_dir = str(prompt_dir)
        
    def generate_sus_prompt(self, encoded_gt_frame_grid):
        print("\n--- [1/4] Task Description ---")
        task_agent = TaskDescriptor(self.cfg, self.prompt_dir)
        task_desc = task_agent.analyse(encoded_gt_frame_grid)
        print("Done.")
        
        print("\n--- [2/4] Contact Analysis ---")
        contact_agent = ContactSequenceAnalyser(self.cfg, self.prompt_dir)
        contact_pattern = contact_agent.analyse(encoded_gt_frame_grid)
        print("Done.")
        
        print("\n--- [3/4] Gait Analysis ---")
        gait_agent = GaitAnalyser(self.cfg, self.prompt_dir)
        gait_response = gait_agent.analyse(encoded_gt_frame_grid, contact_pattern)
        print("Done.")
        
        print("\n--- [4/4] Physics Requirements ---")
        req_agent = TaskRequirementAnalyser(self.cfg, self.prompt_dir)
        req_response = req_agent.analyse(encoded_gt_frame_grid)
        print("Done.")
        
        # 종합
        self.prepare_user_content([
            {"type":"text", "data": f"Task Description: {task_desc}"},
            {"type":"text", "data": f"Contact Pattern: {contact_pattern}"},
            {"type":"text", "data": f"Gait Analysis: {gait_response}"},
            {"type":"text", "data": f"Physics Requirements: {req_response}"}
        ])
        
        print("\n--- Generating Final SUS Report ---")
        sus_prompt = self.query()
        return sus_prompt
