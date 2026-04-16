import re
from collections import Counter, defaultdict

class CompanyNameCleaner:
    def __init__(self, en_to_ko_map, typo_fix_map, conflict_groups, protected_keywords):
      # 전처리 관련 custom filter들
      self.en_to_ko_map = en_to_ko_map
      self.typo_fix_map = typo_fix_map
      self.conflict_groups = conflict_groups
      self.protected_keywords = protected_keywords

      #추후 fit으로 계산할것
      self.counts = None
      self.name_to_conflict_words = None
      self.correction_map = None
      
    def _precompute_conflicts(self):
      """각 이름이 어떤 충돌 그룹의 단어를 가지고 있는지 미리 계산"""
      mapping = defaultdict(set)
      all_unique_names = self.counts.keys()

      # 모든 충돌 단어들을 하나의 세트로 합침
      all_conflict_words = set().union(*self.conflict_groups)

      for name in all_unique_names:
        name_low = name.lower()
        for word in all_conflict_words:
          if word in name_low:
            mapping[name].add(word)
      return mapping

    def fit(self, company_series):
      """Company series를 받아 사전 계산을 수행합니다."""
      # [1차 정제] 기초 정규화 수행
      raw_series = company_series.apply(self.basic_normalize)

      # 1차 정제 이후 각 회사명의 등장 횟수
      self.counts = Counter(raw_series)

      # [성능 최적화] 각 회사명이 어떤 충돌 키워드를 포함하는지 계산
      # 이후 2차 정규화 시 O(1)로 연산 수행 가능
      self.name_to_conflict_words = self._precompute_conflicts()

      # [2차 정규화] 추가 교정 맵 생성
      self.correction_map = self.build_correction_map()

    def basic_normalize(self, name):
      """기초적인 정규화를 수행합니다."""
      # 문자열이 아니거나 빈 문자열이 들어온경우
      if not isinstance(name, str) or name.strip() == "": return ""

      # 1. 괄호 + 괄호 안 내용 제거
      name = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}|<.*?>", "", name)

      # 2. 명백한 오타 (팬다 -> 판다 등) 교정
      for typo, correct in self.typo_fix_map.items():
        name = name.replace(typo, correct)

      # 3. 영문 변환 (Solutions 등 영문 키워드를 한글로 번역)
      for en_pattern, ko_word in self.en_to_ko_map.items():
        name = re.sub(en_pattern, ko_word, name, flags=re.IGNORECASE)

      # 4. 접미사 통일 (표기법 단일화)
      name = re.sub(r"솔루션[스즈]?$|솔루션[스즈]?(?=\s)", "솔루션즈", name)
      name = re.sub(r"네트웍[스즈]?$|네트워크$", "네트웍스", name)
      name = re.sub(r"시스템[스즈]?$", "시스템즈", name)
      name = re.sub(r"게임[스즈]?$", "게임즈", name)

      # 공백 처리: 연속된 공백을 하나로 합치고 양 끝 공백 제
      name = re.sub(r"\s+", " ", name)
      return name.strip()

    def is_edit_distance_one(self, s1, s2):
      """두 단어의 edit distance가 1인지 판별합니다."""
      l1, l2 = len(s1), len(s2)
      if abs(l1 - l2) > 1: return False
      if l1 < l2: s1, s2 = s2, s1
      for i in range(len(s2)):
        if s1[i] != s2[i]:
          return s1[i+1:] == s2[i+1:] if l1 == l2 else s1[i+1:] == s2[i:]
      return True

    def build_correction_map(self):
      # 1. 길이에 따라 미리 정렬 (길이 차이 1 이내만 비교하기 위해)
      unique_names = sorted(list(self.counts.keys()), key=len)
      mapping = {}

      for i in range(len(unique_names)):
        n1 = unique_names[i]
        for j in range(i + 1, len(unique_names)):
          n2 = unique_names[j]

        # [최적화] 길이가 2 이상 차이 나면 더 이상 볼 필요 없음 (정렬 덕분)
        if len(n2) - len(n1) > 1:
          break

        # [최적화] 예외 및 보호 키워드는 건너뜀
        if n1 in self.protected_keywords or n2 in self.protected_keywords: continue

        # [최적화 3] 인덱싱된 충돌 키워드 체크
        n1_words = self.name_to_conflict_words[n1]
        n2_words = self.name_to_conflict_words[n2]

        if n1_words and n2_words:
          is_conflict = False
          for group in self.conflict_groups:
            c1 = n1_words.intersection(group)
            c2 = n2_words.intersection(group)
            if c1 and c2 and c1 != c2:
              is_conflict = True
              break
          if is_conflict: continue

        # 4. 편집 거리 체크
        if self.is_edit_distance_one(n1, n2):
          major = n1 if self.counts[n1] >= self.counts[n2] else n2
          minor = n2 if major == n1 else n1
          mapping[minor] = major # 빈도 수 많은것으로 맞추기
      return mapping

    def clean(self, name):
      s1 = self.basic_normalize(name)
      s2 = self.correction_map.get(s1, s1)
      return self.basic_normalize(s2)