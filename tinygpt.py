"""
TinyGPT 실험 프로젝트
==================
GPT의 작동 방식을 이해하기 위한 간단한 실험
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
import argparse
import os
import sys
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('tinygpt')

# 명령행 인자 설정
def setup_argparser():
    """명령행 인자 설정 함수"""
    parser = argparse.ArgumentParser(description='TinyGPT 실험')
    parser.add_argument('--block_size', type=int, default=8, 
                        help='문맥 범위 크기 (기본값: 8)')
    parser.add_argument('--learning_mode', choices=['quick', 'normal', 'deep'], default='normal',
                        help='학습 모드 (quick: 100스텝, normal: 500스텝, deep: 2000스텝)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='텍스트 생성 시 온도 설정 (기본값: 0.5, 높을수록 창의적)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='학습률 (기본값: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--no_interactive', action='store_true',
                        help='대화형 모드 비활성화')
    parser.add_argument('--text_file', type=str, default='text.txt',
                        help='학습 데이터 파일 경로 (기본값: text.txt)')
    return parser

# 시드 설정 함수
def set_seed(seed):
    """랜덤 시드 설정"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"랜덤 시드 설정: {seed}")

# 실험 ID 생성 함수
def get_experiment_id():
    """고유한 실험 ID 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 디바이스 설정
def get_device():
    """사용 가능한 디바이스 감지"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CPU 사용")
    return device

# 멀티헤드 어텐션 클래스
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        assert embed_size % num_heads == 0, "임베딩 크기는 헤드 수로 나누어 떨어져야 합니다"
        
        self.head_size = embed_size // num_heads
        
        # 키, 쿼리, 밸류 변환 레이어
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        
        # 출력 프로젝션
        self.proj = nn.Linear(embed_size, embed_size)
        
        # 마스크: 미래 토큰 볼 수 없게 함 (인과적 어텐션)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        B, T, C = x.shape  # 배치, 타임 스텝(시퀀스 길이), 채널(임베딩 차원)
        
        # 키, 쿼리, 밸류 계산 및 헤드 나누기
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # 어텐션 계산: Q와 K의 내적, 스케일링, 마스킹, 소프트맥스, 드롭아웃
        # Q, K의 행렬곱: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # 마스킹
        att = F.softmax(att, dim=-1)  # 소프트맥스
        att = self.dropout(att)  # 드롭아웃
        
        # 어텐션 값과 V의 행렬곱
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ v
        
        # 헤드 합치기
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # 최종 프로젝션
        out = self.proj(out)  # (B, T, C)
        
        return out

# 트랜스포머 블록 클래스
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        # 레이어 정규화
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        
        # 멀티헤드 어텐션
        self.attention = MultiHeadAttention(embed_size, num_heads, block_size)
        
        # 피드포워드 네트워크
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # 첫 번째 서브레이어: 멀티헤드 어텐션 (잔차 연결 포함)
        x = x + self.attention(self.layer_norm1(x))
        # 두 번째 서브레이어: MLP (잔차 연결 포함)
        x = x + self.mlp(self.layer_norm2(x))
        return x

# TinyGPT 모델 클래스
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=8):
        super().__init__()
        self.block_size = block_size
        
        # 문자 임베딩 레이어
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        # 위치 임베딩 레이어
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embed_size))
        self.dropout = nn.Dropout(0.2)
        
        # 트랜스포머 블록
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size) for _ in range(num_layers)
        ])
        
        # 최종 레이어 정규화
        self.layer_norm = nn.LayerNorm(embed_size)
        # 출력 레이어
        self.output_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, idx):
        B, T = idx.shape  # 배치 크기, 시퀀스 길이
        
        # 토큰 임베딩과 위치 임베딩
        token_embed = self.token_embedding(idx)  # (B, T, C)
        position_embed = self.position_embedding[:, :T, :]  # (1, T, C)
        x = self.dropout(token_embed + position_embed)
        
        # 트랜스포머 블록 통과
        x = self.blocks(x)
        x = self.layer_norm(x)
        
        # 출력 레이어 (각 위치에서 다음 토큰 예측)
        logits = self.output_head(x)  # (B, T, vocab_size)
        
        return logits

    def save(self, filepath, vocab_info):
        """모델 저장 메서드"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab': vocab_info['vocab'],
            'char_to_index': vocab_info['char_to_index'],
            'index_to_char': vocab_info['index_to_char'],
            'block_size': self.block_size,
            'model_config': {
                'vocab_size': len(vocab_info['vocab']),
                'embed_size': self.token_embedding.weight.shape[1],
                'num_layers': len(self.blocks),
                'num_heads': 4,  # 현재 고정된 값
            }
        }, filepath)
        logger.info(f"모델 저장 완료: {filepath}")

    @classmethod
    def load(cls, filepath, device=torch.device('cpu')):
        """모델 로드 클래스 메서드"""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            # block_size 확인 및 설정
            if 'block_size' not in checkpoint:
                logger.warning(f"주의: {filepath}에 block_size 정보가 없습니다.")
                # 기본값 사용
                checkpoint['block_size'] = 8
                logger.warning(f"   기본값 8을 사용합니다.")
            
            model_config = checkpoint.get('model_config', {
                'vocab_size': len(checkpoint['vocab']),
                'block_size': checkpoint['block_size'],
            })
            
            # 모델 초기화
            model = cls(
                vocab_size=model_config['vocab_size'],
                block_size=checkpoint['block_size']
            )
            
            # 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 어휘 정보 로드
            vocab_info = {
                'vocab': checkpoint['vocab'],
                'char_to_index': checkpoint['char_to_index'],
                'index_to_char': checkpoint['index_to_char'],
                'block_size': checkpoint['block_size']
            }
            
            return model, vocab_info
        except Exception as e:
            logger.error(f"모델 로드 오류: {str(e)}")
            return None, None

# 데이터 관련 함수들
class TextDataHandler:
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.text = None
        self.chars = None
        self.vocab_size = None
        self.char_to_index = None
        self.index_to_char = None
        self.data = None
        self.train_data = None
        self.val_data = None
    
    def load_text(self):
        """텍스트 파일 로드"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
            logger.info(f"학습 데이터 로드 완료: {self.file_path}")
            logger.info(f"   - 총 {len(self.text)} 글자")
            logger.info(f"   - 약 {len(self.text.split('.'))} 문장")
            return True
        except FileNotFoundError:
            logger.warning(f"주의: {self.file_path} 파일을 찾을 수 없습니다. 예제 텍스트를 생성합니다.")
            self.text = "안녕하세요. 이것은 예제 텍스트입니다. 더 많은 학습 데이터를 추가해주세요."
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(self.text)
            return True
        except Exception as e:
            logger.error(f"텍스트 로드 오류: {str(e)}")
            return False
    
    def process_text(self):
        """텍스트 처리 및 인코딩 준비"""
        # 고유 문자 목록 생성
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        logger.info(f"   - 사용된 고유 문자: {self.vocab_size}개")
        
        # 인코딩/디코딩 사전 생성
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 전체 텍스트를 숫자로 변환
        encoded = self.encode(self.text)
        self.data = torch.tensor(encoded, dtype=torch.long)
        
        # 학습/검증 데이터 분리 (90%/10%)
        self.train_data = self.data[:int(0.9*len(self.data))]
        self.val_data = self.data[int(0.9*len(self.data)):]
        logger.info(f"   - 학습 데이터: {len(self.train_data)} 토큰")
        logger.info(f"   - 검증 데이터: {len(self.val_data)} 토큰")
        
        if len(self.train_data) <= self.block_size:
            logger.warning(f"주의: 데이터 길이({len(self.train_data)})가 블록 크기({self.block_size})보다 작습니다.")
            logger.warning("더 많은 데이터를 추가하거나 블록 크기를 줄이는 것을 고려하세요.")
        
        return {
            'vocab': self.chars,
            'char_to_index': self.char_to_index,
            'index_to_char': self.index_to_char
        }
    
    def encode(self, s):
        """문자열을 숫자 리스트로 변환"""
        return [self.char_to_index.get(c, 0) for c in s]
    
    def decode(self, l):
        """숫자 리스트를 문자열로 변환"""
        return ''.join([self.index_to_char[i] for i in l])
    
    def get_batch(self, split, batch_size):
        """데이터 배치 생성 함수"""
        data = self.train_data if split == 'train' else self.val_data
        if len(data) <= self.block_size:
            raise ValueError(f"데이터 길이({len(data)})가 블록 크기({self.block_size})보다 작습니다.")
        
        # 랜덤한 시작 위치 선택
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        # 입력 시퀀스 (X)
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        # 타겟 시퀀스 (Y): 입력보다 한 칸 뒤의 글자들
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

# 텍스트 생성 함수
def generate_sample(model, context, vocab_info, device, max_new_tokens=30, temp=0.5):
    """모델로 텍스트 생성"""
    char_to_index = vocab_info['char_to_index']
    index_to_char = vocab_info['index_to_char']
    
    # block_size 키가 없을 경우 모델의 block_size 사용
    if 'block_size' in vocab_info:
        block_size = vocab_info['block_size']
    else:
        block_size = model.block_size
        # 오류 없이 계속 진행하기 위해 vocab_info에 block_size 추가
        vocab_info['block_size'] = block_size
        logger.warning(f"주의: vocab_info에 block_size가 없어 모델의 값({block_size})을 사용합니다.")
    
    # 입력 텍스트를 인덱스로 변환
    idx = torch.tensor([char_to_index.get(s, 0) for s in context], dtype=torch.long).unsqueeze(0).to(device)
    
    # 입력에 없는 문자 처리
    unknown_chars = [c for c in context if c not in char_to_index]
    if unknown_chars:
        logger.warning(f"주의: '{', '.join(unknown_chars)}' 문자는 학습 데이터에 없어 무시됩니다.")
    
    generated_text = context
    
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 입력이 블록 크기보다 길면 자르기
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
                
            # 모델을 통해 다음 토큰 예측
            logits = model(idx)
            logits = logits[:, -1, :]  # 마지막 위치의 예측만 사용
            
            # 온도 조절 샘플링
            probs = F.softmax(logits / temp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰 추가
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 생성된 문자 추가
            next_char = index_to_char[idx_next.item()]
            generated_text += next_char
            
            # 문장이 끝나면 줄바꿈
            if next_char == '.':
                generated_text += '\n'
                
    model.train()  # 학습 모드로 복귀
    return generated_text

# 모델 학습 함수
def train_model(model, data_handler, device, args, experiment_id):
    """모델 학습 함수"""
    logger.info("\n학습 시작...")
    
    # 결과 폴더 생성
    results_dir = f"results/{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"결과 저장 디렉토리: {results_dir}")
    
    # 실험 설정 저장
    with open(f"{results_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    
    # 학습 관련 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 학습 상태 변수
    losses = []
    best_loss = float('inf')
    
    # 샘플 생성 지점
    sample_points = {
        'early': int(args.total_steps * 0.05),  # 5%
        'mid': int(args.total_steps * 0.5),     # 50%
        'final': args.total_steps - 1           # 마지막
    }
    samples = {phase: None for phase in sample_points}
    
    start_time = time.time()
    
    try:
        for step in range(args.total_steps):
            # 배치 가져오기
            xb, yb = data_handler.get_batch('train', args.batch_size)
            xb, yb = xb.to(device), yb.to(device)
            
            # 예측 및 손실 계산
            logits = model(xb)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = yb.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
            # 손실 기록
            losses.append(loss.item())
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 샘플 생성 지점 확인
            for phase, phase_step in sample_points.items():
                if step == phase_step:
                    vocab_info = {
                        'char_to_index': data_handler.char_to_index,
                        'index_to_char': data_handler.index_to_char,
                        'block_size': data_handler.block_size
                    }
                    samples[phase] = generate_sample(model, "나는 ", vocab_info, device, 30, args.temperature)
            
            # 진행 상황 출력 (10% 단위)
            progress_interval = max(1, args.total_steps // 10)
            if step % progress_interval == 0 or step == args.total_steps - 1:
                progress = step / args.total_steps * 100
                elapsed = time.time() - start_time
                
                # 1000스텝당 시간 계산
                steps_per_sec = (step + 1) / elapsed
                time_per_1k_steps = 1000 / steps_per_sec if steps_per_sec > 0 else 0
                
                logger.info(f"진행률: {progress:.1f}% | 스텝 {step+1}/{args.total_steps} | " 
                           f"손실: {loss.item():.4f} | "
                           f"최저 손실: {best_loss:.4f} | "
                           f"시간: {elapsed:.1f}초 (1K스텝당 약 {time_per_1k_steps:.1f}초)")
            
            # 20스텝마다 검증 및 모델 저장
            if step % 20 == 0:
                # 검증 데이터에서 손실 계산
                try:
                    val_x, val_y = data_handler.get_batch('val', args.batch_size)
                    val_x, val_y = val_x.to(device), val_y.to(device)
                    with torch.no_grad():
                        val_logits = model(val_x)
                        val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                        
                        # 현재 검증 손실이 최저 손실보다 낮으면 최저 손실 갱신 및 모델 저장
                        if val_loss.item() < best_loss:
                            best_loss = val_loss.item()
                            
                            # 최고 성능 모델 저장
                            vocab_info = {
                                'vocab': data_handler.chars,
                                'char_to_index': data_handler.char_to_index,
                                'index_to_char': data_handler.index_to_char,
                                'block_size': data_handler.block_size
                            }
                            model.save(f"{results_dir}/best_model.pt", vocab_info)
                except Exception as e:
                    logger.warning(f"검증 중 오류 발생: {str(e)}")
    except KeyboardInterrupt:
        logger.info("\n학습 중단: 사용자에 의한 중단")
    
    # 학습 완료 시간 및 총 소요 시간
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\n총 학습 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")
    
    # 최종 모델 저장
    vocab_info = {
        'vocab': data_handler.chars,
        'char_to_index': data_handler.char_to_index,
        'index_to_char': data_handler.index_to_char,
        'block_size': data_handler.block_size
    }
    model.save(f"{results_dir}/final_model.pt", vocab_info)
    
    # 손실 그래프 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Learning Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (lower is better)')
    plt.grid(True)
    plt.savefig(f"{results_dir}/learning_curve.png")
    
    logger.info("\n학습 완료!")
    
    # 최종 샘플 생성
    if samples['final'] is None:  # 아직 생성되지 않았다면
        vocab_info = {
            'char_to_index': data_handler.char_to_index,
            'index_to_char': data_handler.index_to_char,
            'block_size': data_handler.block_size
        }
        samples['final'] = generate_sample(model, "나는 ", vocab_info, device, 30, args.temperature)
    
    # 학습 단계별 생성 결과 비교
    logger.info("\n학습 진행에 따른 텍스트 생성 결과 비교:")
    
    for phase, sample in samples.items():
        if sample:
            phase_name = {
                'early': '학습 초기',
                'mid': '학습 중간',
                'final': '학습 완료'
            }[phase]
            logger.info(f"\n[{phase_name}]")
            logger.info(sample)
    
    return {
        'model': model,
        'results_dir': results_dir,
        'samples': samples,
        'best_loss': best_loss,
        'vocab_info': vocab_info
    }

# 대화형 실험 모드
def interactive_experiment(model, vocab_info, device, temperature=0.5):
    """대화형 실험 모드"""
    logger.info("\n대화형 실험 모드 시작 (종료: 'q' 입력)")
    
    while True:
        # 사용자 입력
        prompt = input("\n시작 텍스트 입력: ")
        if prompt.lower() == 'q':
            break
            
        # 온도 입력
        temp_input = input("온도 설정 (0.1-2.0, 기본값 0.5): ")
        try:
            temp = float(temp_input) if temp_input.strip() else temperature
            if temp < 0.1:
                temp = 0.1
                logger.info("온도가 0.1 미만이어서 0.1로 설정되었습니다.")
            elif temp > 2.0:
                temp = 2.0
                logger.info("온도가 2.0 초과이어서 2.0으로 설정되었습니다.")
        except:
            temp = temperature
            logger.info(f"유효하지 않은 온도 값이어서 기본값 {temperature}로 설정되었습니다.")
        
        # 생성 길이 입력
        length_input = input("생성할 글자 수 (기본값 40): ")
        try:
            length = int(length_input) if length_input.strip() else 40
            if length < 1:
                length = 1
                logger.info("생성 길이가 1 미만이어서 1로 설정되었습니다.")
            elif length > 500:
                length = 500
                logger.info("생성 길이가 500을 초과하여 500으로 제한되었습니다.")
        except:
            length = 40
            logger.info("유효하지 않은 생성 길이이어서 기본값 40으로 설정되었습니다.")
            
        # 텍스트 생성
        try:
            start_time = time.time()
            result = generate_sample(model, prompt, vocab_info, device, length, temp)
            generation_time = time.time() - start_time
            
            logger.info("\n생성된 텍스트:")
            logger.info(result)
            logger.info(f"생성 시간: {generation_time:.2f}초 (초당 약 {length/generation_time:.1f}자)")
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {str(e)}")

# 메인 함수
def main():
    """메인 함수"""
    # 인자 파서 설정
    parser = setup_argparser()
    args = parser.parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 실험 ID 생성
    experiment_id = get_experiment_id()
    logger.info(f"실험 ID: {experiment_id}")
    
    # 디바이스 설정
    device = get_device()
    
    # 학습 모드에 따른 스텝 수 설정
    if args.learning_mode == 'quick':
        args.total_steps = 100
        logger.info("빠른 학습 모드: 100 스텝")
    elif args.learning_mode == 'deep':
        args.total_steps = 2000
        logger.info("깊은 학습 모드: 2000 스텝")
    else:  # normal
        args.total_steps = 500
        logger.info("일반 학습 모드: 500 스텝")
    
    # 설정 정보 출력
    logger.info(f"\n실험 설정: 문맥 범위(block_size) = {args.block_size}, 온도 = {args.temperature}")
    logger.info(f"   - 배치 크기: {args.batch_size}, 학습률: {args.learning_rate}")
    
    # 데이터 처리
    data_handler = TextDataHandler(args.text_file, args.block_size)
    if not data_handler.load_text():
        logger.error("텍스트 데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    vocab_info = data_handler.process_text()
    
    # 모델 초기화
    logger.info("\nTinyGPT 모델 초기화 중...")
    model = TinyGPT(
        vocab_size=data_handler.vocab_size,
        embed_size=128,
        num_heads=4,
        num_layers=4,
        block_size=args.block_size
    ).to(device)
    
    logger.info(f"   - 어휘 크기: {data_handler.vocab_size}")
    logger.info(f"   - 모델 구성: 임베딩 128차원, 4개 레이어, 4개 헤드")
    logger.info(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 모델 학습
    train_results = train_model(model, data_handler, device, args, experiment_id)
    
    # 추가 실험
    logger.info("\n추가 실험 시작:")
    
    # 다양한 시작 프롬프트로 생성 실험
    prompts = ["나는 ", "고양이는 ", "사랑은 ", "인공지능이 "]
    logger.info("\n다양한 시작 프롬프트 실험:")
    
    # train_results['vocab_info']에 block_size 확인
    if 'block_size' not in train_results['vocab_info']:
        logger.warning("vocab_info에 block_size가 없습니다. 추가합니다.")
        train_results['vocab_info']['block_size'] = model.block_size
    
    for prompt in prompts:
        logger.info(f"\n[프롬프트: '{prompt}']")
        logger.info(generate_sample(
            model, 
            prompt, 
            train_results['vocab_info'], 
            device, 
            40, 
            args.temperature
        ))
    
    # 온도 변화 실험
    logger.info("\n온도 변화 실험 (같은 프롬프트, 다른 온도):")
    temperatures = [0.3, 0.7, 1.2]
    for temp in temperatures:
        logger.info(f"\n[온도: {temp}]")
        logger.info(generate_sample(
            model, 
            "나는 ", 
            train_results['vocab_info'], 
            device, 
            40, 
            temp
        ))
    
    # 대화형 실험 모드
    if not args.no_interactive:
        interactive_experiment(
            model, 
            train_results['vocab_info'], 
            device, 
            args.temperature
        )
    else:
        logger.info("\n대화형 모드가 비활성화되었습니다.")

# 프로그램 시작점
if __name__ == "__main__":
    main() 