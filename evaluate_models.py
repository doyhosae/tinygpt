"""
TinyGPT 모델 평가 도구
=====================
best_model.pt와 final_model.pt를 비교하고 평가하는 스크립트
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import shutil
from datetime import datetime

# 명령행 인자 설정
parser = argparse.ArgumentParser(description='TinyGPT 모델 평가')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='텍스트 생성 시 온도 설정 (기본값: 0.7)')
parser.add_argument('--num_tests', type=int, default=3,
                    help='각 프롬프트당 테스트 횟수 (기본값: 3)')
parser.add_argument('--max_tokens', type=int, default=30,
                    help='생성할 최대 토큰 수 (기본값: 30)')
args = parser.parse_args()

# TinyGPT 모델 클래스 (tinygpt.py에서 가져옴)
# 원래 tinygpt.py에서 필요한 클래스와 함수를 임포트하는 것이 좋지만,
# 여기서는 독립적으로 실행할 수 있도록 필요한 클래스를 다시 정의합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        
        self.proj = nn.Linear(embed_size, embed_size)
        
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        
        self.attention = MultiHeadAttention(embed_size, num_heads, block_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.2),
        )
        
    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=8):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embed_size))
        self.dropout = nn.Dropout(0.2)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_size, num_heads, block_size) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_size)
        self.output_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, idx):
        B, T = idx.shape
        
        token_embed = self.token_embedding(idx)
        position_embed = self.position_embedding[:, :T, :]
        x = self.dropout(token_embed + position_embed)
        
        x = self.blocks(x)
        x = self.layer_norm(x)
        
        logits = self.output_head(x)
        
        return logits

# 텍스트 생성 함수
def generate_sample(model, context, char_to_index, index_to_char, block_size, max_new_tokens=30, temp=0.7):
    """현재 모델로 텍스트 생성"""
    idx = torch.tensor([char_to_index.get(s, 0) for s in context], dtype=torch.long).unsqueeze(0)
    
    unknown_chars = [c for c in context if c not in char_to_index]
    if unknown_chars:
        print(f"\n주의: '{', '.join(unknown_chars)}' 문자는 학습 데이터에 없어 무시됩니다.")
    
    generated_text = context
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
                
            logits = model(idx)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits / temp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            next_char = index_to_char[idx_next.item()]
            generated_text += next_char
            
            if next_char == '.':
                generated_text += '\n'
                
    return generated_text

def load_model(model_path):
    """모델 로드 함수"""
    if not os.path.exists(model_path):
        print(f"오류: {model_path} 파일이 존재하지 않습니다.")
        return None
    
    try:
        saved_data = torch.load(model_path)
        
        # block_size 확인
        if 'block_size' not in saved_data:
            print(f"주의: {model_path}에 block_size 정보가 없습니다.")
            # 기본 block_size 사용
            block_size = 8
            saved_data['block_size'] = block_size
            print(f"   기본값 {block_size}를 사용합니다.")
        
        # 모델 재생성
        model = TinyGPT(
            vocab_size=len(saved_data['vocab']),
            block_size=saved_data['block_size']
        )
        model.load_state_dict(saved_data['model_state_dict'])
        
        return {
            'model': model,
            'vocab': saved_data['vocab'],
            'char_to_index': saved_data['char_to_index'],
            'index_to_char': saved_data['index_to_char'],
            'block_size': saved_data['block_size']
        }
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

def evaluate_text(text, criteria):
    """텍스트 평가 (수동 입력)"""
    print("\n" + "=" * 40)
    print(f"생성된 텍스트 평가: ")
    print("-" * 40)
    print(text)
    print("-" * 40)
    
    scores = {}
    for criterion in criteria:
        while True:
            try:
                score = int(input(f"{criterion} 점수 (1-5): "))
                if 1 <= score <= 5:
                    scores[criterion] = score
                    break
                else:
                    print("주의: 1에서 5 사이의 점수를 입력해주세요.")
            except ValueError:
                print("주의: 숫자를 입력해주세요.")
    
    return scores

def main():
    # 평가 기준
    criteria = ['일관성', '자연스러움', '문맥 유지', '다양성']
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요 ",
        "오늘 날씨가 ",
        "인공지능은 ",
        "우리나라의 ",
        "음식 중에서 "
    ]
    
    print("TinyGPT 모델 평가 도구")
    print(f"설정: 온도={args.temperature}, 테스트 횟수={args.num_tests}, 최대 토큰={args.max_tokens}")
    
    # 결과 저장 디렉토리
    results_dir = "evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"\n평가 결과 저장 디렉토리 생성: {results_dir}")
    
    # 모델 로드
    print("\n모델 로드 중...")
    best_model_data = load_model("results/best_model.pt")
    final_model_data = load_model("results/final_model.pt")
    
    if not best_model_data or not final_model_data:
        print("모델 로드 실패. 프로그램을 종료합니다.")
        return
    
    # 평가 결과 저장 변수
    results = {
        'best_model': {'총점': 0, '평균': 0, '세부 점수': {}},
        'final_model': {'총점': 0, '평균': 0, '세부 점수': {}}
    }
    
    # 각 프롬프트에 대해 평가 수행
    for prompt in test_prompts:
        print(f"\n\n{'='*50}")
        print(f"시작 텍스트: '{prompt}'")
        
        results['best_model']['세부 점수'][prompt] = []
        results['final_model']['세부 점수'][prompt] = []
        
        # 여러 번 테스트
        for i in range(args.num_tests):
            print(f"\n테스트 #{i+1}")
            
            # best_model 텍스트 생성
            best_text = generate_sample(
                best_model_data['model'], 
                prompt, 
                best_model_data['char_to_index'],
                best_model_data['index_to_char'],
                best_model_data['block_size'],
                args.max_tokens, 
                args.temperature
            )
            
            print("\n[최고 성능 모델]")
            best_scores = evaluate_text(best_text, criteria)
            results['best_model']['세부 점수'][prompt].append({
                '텍스트': best_text,
                '점수': best_scores
            })
            
            # final_model 텍스트 생성
            final_text = generate_sample(
                final_model_data['model'], 
                prompt, 
                final_model_data['char_to_index'],
                final_model_data['index_to_char'],
                final_model_data['block_size'],
                args.max_tokens, 
                args.temperature
            )
            
            print("\n[최종 모델]")
            final_scores = evaluate_text(final_text, criteria)
            results['final_model']['세부 점수'][prompt].append({
                '텍스트': final_text,
                '점수': final_scores
            })
    
    # 결과 집계
    for model_name in ['best_model', 'final_model']:
        total_score = 0
        total_criteria = 0
        
        for prompt in results[model_name]['세부 점수']:
            for test in results[model_name]['세부 점수'][prompt]:
                for criterion in test['점수']:
                    total_score += test['점수'][criterion]
                    total_criteria += 1
        
        results[model_name]['총점'] = total_score
        results[model_name]['평균'] = total_score / total_criteria if total_criteria > 0 else 0
    
    # 결과 출력
    print("\n\n" + "="*50)
    print("평가 결과 요약")
    print("="*50)
    print(f"최고 성능 모델 (best_model): 총점 {results['best_model']['총점']}, 평균 {results['best_model']['평균']:.2f}/5")
    print(f"최종 모델 (final_model): 총점 {results['final_model']['총점']}, 평균 {results['final_model']['평균']:.2f}/5")
    
    # 우승 모델 결정
    winner = "best_model" if results['best_model']['평균'] >= results['final_model']['평균'] else "final_model"
    print(f"\n추천 모델: {winner} (평균 점수: {results[winner]['평균']:.2f}/5)")
    
    # 선택한 모델을 최종 모델로 복사
    model_path = "results/best_model.pt" if winner == "best_model" else "results/final_model.pt"
    save_path = "results/model.pt"
    
    copy_model = input(f"\n{winner}를 model.pt로 저장하시겠습니까? (y/n): ")
    if copy_model.lower() == 'y':
        shutil.copy2(model_path, save_path)
        print(f"{winner}를 {save_path}로 복사했습니다.")
    
    # 평가 결과 시각화
    plt.figure(figsize=(10, 6))
    models = ['최고 성능 모델', '최종 모델']
    scores = [results['best_model']['평균'], results['final_model']['평균']]
    
    plt.bar(models, scores, color=['royalblue', 'lightcoral'])
    plt.ylim(0, 5)
    plt.title('모델 평균 성능 비교')
    plt.ylabel('평균 점수 (1-5)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(scores):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{results_dir}/model_comparison_{timestamp}.png"
    plt.savefig(plot_path)
    
    print(f"\n비교 그래프 저장 완료: {plot_path}")
    print("\n평가 완료!")

if __name__ == "__main__":
    main() 