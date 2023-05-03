import pdfplumber
import re
import inspect
import copy
import os
import shutil

class Parser(object):
    def __init__(self):
        # 텍스트 spacing을 위한 구분자 튜플들.
        self.josa = ('이', '가', '께서', '에서', '어서', '에게서', '의', '을', '를', '에', '에게', '께', '한테', '로서', '로써', '고', '라고', '와', '과', '랑', '같이', '처럼', '만큼', '만치','하고', '이며', '에다', '에다가', '랑', '은', '는', '도', '부터', '까지', '마저', '조차', '만', '따라', '따라서', '토록', '치고', '대로', '나', '란', '든가', '든지', '나마', '로', '으로', '는데', '며', '되어', '위해', '대해', '한', '하지', '해', '하여', '인', '된', '각각', '음', '함', '하기', '될', '할', '매우') # 조사.
        self.roma_num = ('I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 'IX.', 'X.', 'XI.', 'XII.') # 로마 숫자.
        self.round_num = ('①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '⑪', '⑫', '⑬', '⑭', '⑮') # 동그라미 숫자.
        self.context_token = ('-', ':', '=', '/') # 한 문맥을 구분하는 토큰 튜플. 거의 '-'와 ':'만 있음.

        self.sentence_end_flag = False # 카테고리 시작 토큰 중 '다.'에 대한 예외 처리를 위한 flag. ('가.', '나.' '다.'와 같이 카테고리 시작 토큰이면 False. / '했습니다.'와 같이 문장이 잘려서 밑에 '다.'로 시작하는 경우의 line이면 True.)

        pdfplumber_path = f'{inspect.getfile(pdfplumber).split("pdfplumber/")[0]}pdfplumber' # pdfplumber 패키지 경로.

        new_page_path = f'{pdfplumber_path}/page.py' # 수정된 pdfplumber page.py 경로.
        new_table_path = f'{pdfplumber_path}/table.py' # 수정된 pdfplumber table.py 경로.
        new_utils_path = f'{pdfplumber_path}/utils.py' # 수정된 pdfplumber utils.py 경로.

        shutil.copy(f'{os.getcwd()}/new_pdfplumber/page.py', new_page_path)
        shutil.copy(f'{os.getcwd()}/new_pdfplumber/table.py', new_table_path)
        shutil.copy(f'{os.getcwd()}/new_pdfplumber/utils.py', new_utils_path)

    # 문서의 table 섹션들을 extract_tables의 결과 table로 교체 & 문서 전체 텍스트 파싱 함수.
    def parsing(self, pdf_file_path: str) -> list:  
        if pdf_file_path:
            pdf_file_path = pdf_file_path
        else:
            raise ValueError('대상 pdf 파일의 이름을 입력해주세요.')

        with pdfplumber.open(pdf_file_path) as pdf:
            past_table_xs = None # 페이지로 인해 테이블 분할된 경우, 이전 테이블의 x0 값들을 저장하는 변수.
            doc_text_list = [] # 문서 전체(모든 페이지)에 대한 텍스트 리스트. 한 페이지의 이전 line을 확인할 때 사용하는 용도. (굳이 리스트로 쌓는 이유는, 페이지가 여러 개일 때 한 페이지의 첫 line 처리 시 이전 페이지의 마지막 line을 past_line으로 접근해야 함.)
            result_list = []

            for page_idx in range(len(pdf.pages)):
                page_data = pdf.pages[page_idx] # 한 페이지에 대한 객체.

                # 한 페이지에서 추출한 table 데이터. (없을 시 empty list 반환)
                page_table_list = page_data.extract_tables(table_settings={
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict"
                }) # lines_strict 옵션으로 설정해야 테이블 내부의 텍스트가 확실히 구분됨.

                page_text_list = [page_text.replace(' ', '').strip() for page_text in page_data.extract_text().split('\n')] # 한 페이지의 line 별 텍스트들에 대한 리스트.
                cover_text = ''.join(page_text_list).replace(' ', '') # 커버 페이지 확인 용도의 텍스트. (띄어쓰기 전부 제거)

                table_count = 0 # 한 페이지의 테이블 순서.
                table_flag = False # 한 페이지 내에서 테이블 섹션 구분하는 flag.
                splited_table_flag = False

                # 한 페이지 내에서 line 순서별 로직 처리.
                for now_line_idx, now_line in enumerate(page_text_list): # now_line: 현재 line의 텍스트.
                    if not table_flag:
                        if now_line == '': # 현재 line이 공백인 경우
                            pass
                        elif '[TABLE_START]' not in now_line: # 일반 텍스트 line.
                            if splited_table_flag: # 분할된 테이블에 대한 일반 텍스트들은 필요 없으므로 pass.
                                if '[TABLE_END]' in now_line:
                                    splited_table_flag = False
                                continue

                            if not result_list: # result_list에 처음으로 텍스트가 들어가는 경우
                                doc_text_list.append(now_line)
                                result_list.append(now_line)
                            else: # doc_text_List에 이미 이전 텍스트가 들어있는 경우
                                past_line = str(doc_text_list[-1]) # 이전 line의 텍스트.
                                doc_text_list.append(now_line)
                                temp = self._check_spacing(result_list, past_line, now_line)
                        elif '[TABLE_START]' in now_line: # 테이블 섹션 시작 위치.
                            target_table_list = self._merged_cell_process(page_table_list[table_count]) # 병합된 셀 처리한 table 리스트.
                            target_table_xs = list(map(lambda x: x.split('.')[0], target_table_list[-1]['xs'])) # 현재 테이블의 모든 x0 값을 지닌 리스트.
                            target_table_list = target_table_list[:-1] # x0에 대한 딕셔너리 부분 제거.
                            
                            if re.match('^\(단위.*원\)$', target_table_list[0][0]): # 테이블 첫 번째 row가 단위에 대한 row인 경우
                                if now_line_idx == 0 and type(past_page_bottom) == list: # 분할된 테이블인 경우
                                    pass
                                else:
                                    target_table_list = target_table_list[1:]

                            # 이전 페이지의 마지막과 현재 페이지의 첫 번째가 테이블인 경우(테이블이 페이지로 인해 분할된 경우)
                            if now_line_idx == 0 and type(past_page_bottom) == list:
                                """
                                past_page_bottom: 이전 페이지의 마지막 table 리스트.
                                target_table_list: 현재 페이지의 첫 번째 table 리스트.
                                """

                                # 현재 테이블의 두 번째 row의 첫 cell이 merged여서 비어있는 경우
                                if len(target_table_list) > 1 and (target_table_list[1][0] == '' or target_table_list[1][0] == '[ROW_MERGED]'):
                                    for i in range(1, len(target_table_list)):
                                        if target_table_list[i][0] == '' or target_table_list[i][0] == '[ROW_MERGED]':
                                            target_table_list[i][0] = past_page_bottom[-1][0]

                                past_table_last_row = past_page_bottom[-1] # 이전 페이지 마지막 테이블의 마지막 row.
                                now_table_first_row = target_table_list[0] # 현재 페이지 첫 번째 테이블의 첫 번째 row.
                                segmented_idx_list = []

                                # 이전 테이블의 열 개수가 현재 테이블의 열 개수보다 많은 경우
                                if len(past_table_last_row) > len(now_table_first_row):
                                    temp_table_xs = copy.deepcopy(target_table_xs)
                                    temp_target_table_xs = copy.deepcopy(target_table_xs)
                                    
                                    for ex_x0 in past_table_xs:
                                        if not temp_target_table_xs: # 현재 테이블이 다 순회돼서 현재 x0 리스트가 비어있는 경우
                                            segmented_idx_list.append(segmented_idx_list[-1])
                                            continue

                                        compared_x0 = temp_target_table_xs[0] # 더 짧은 현재 테이블의 처음 x0 값.
                                        if int(ex_x0) in list(range(int(compared_x0) - 4, int(compared_x0) + 4)):
                                            segmented_idx_list.append(temp_table_xs.index(compared_x0))
                                            temp_target_table_xs.pop(0) # 첫 번째 요소 pop.
                                        else: # 추가된 x0인 경우
                                            segmented_idx_list.append(segmented_idx_list[-1])

                                    # 현재 페이지 첫 번째 테이블의 column 개수를 이전 페이지 마지막 테이블 column 개수에 맞게 세분화.
                                    target_table_list = [[target_table_row[segmented_x0_idx] for segmented_x0_idx in segmented_idx_list] for target_table_row in target_table_list]
                                    now_table_first_row = target_table_list[0] # 현재 페이지 첫 번째 테이블의 첫 번째 row 재설정.
                                # 이전 테이블의 열 개수가 현재 테이블의 열 개수보다 적은 경우
                                elif len(past_table_last_row) < len(now_table_first_row):
                                    temp_table_xs = copy.deepcopy(past_table_xs)

                                    for ex_x0 in target_table_xs:
                                        if not past_table_xs: # 이전 테이블이 다 순회돼서 이전 x0 리스트가 비어있는 경우
                                            segmented_idx_list.append(segmented_idx_list[-1])
                                            continue

                                        compared_x0 = past_table_xs[0] # 더 짧은 테이블의 처음 x0 값.
                                        if int(ex_x0) in list(range(int(compared_x0) - 4, int(compared_x0) + 4)):
                                            segmented_idx_list.append(temp_table_xs.index(compared_x0))
                                            past_table_xs.pop(0) # 첫 번째 요소 pop.
                                        else: # 추가된 x0인 경우
                                            segmented_idx_list.append(segmented_idx_list[-1])

                                    # 이전 페이지 마지막 테이블의 column 개수를 현재 페이지 첫 번째 테이블 column 개수에 맞게 세분화.
                                    past_page_bottom = [[past_table_row[segmented_x0_idx] for segmented_x0_idx in segmented_idx_list] for past_table_row in past_page_bottom]
                                    past_table_last_row = past_page_bottom[-1] # 이전 페이지 마지막 테이블의 마지막 row 재설정.

                                # 이전 테이블의 마지막 row와 현재 테이블의 첫 번째 row를 합쳐야 하는 경우
                                if (past_table_last_row[0] != '' and now_table_first_row[0] == '') or (past_table_last_row[0] == '' and now_table_first_row[0] != ''):
                                    past_page_bottom[-1] = [f'{past_table_last_row[i]} {now_table_first_row[i]}' if now_table_first_row[i] != '' else f'{past_table_last_row[i]}{now_table_first_row[i]}' for i in range(len(now_table_first_row))] # 이전 테이블 마지막 row 요소에 현재 테이블 첫 번째 row 요소 합치기.
                                    past_page_bottom.extend(target_table_list[1:])
                                # 안 합쳐도 되는 경우
                                else:
                                    past_page_bottom.extend(target_table_list)

                                doc_text_list[-1] = past_page_bottom
                                result_list[-1] = past_page_bottom
                                
                                table_count += 1
                                table_flag = False

                                if '[TABLE_START]' in now_line and '[TABLE_END]' in now_line: # 분할된 테이블 중 현재 테이블 row가 한 개인 경우
                                    splited_table_flag = False
                                else: # 분할된 테이블 중 현재 테이블 row가 여러 개인 경우
                                    splited_table_flag = True

                                past_table_xs = target_table_xs # 현재 테이블의 xs 값들을 이전 테이블의 xs 값들로 저장.
                                continue
                            
                            doc_text_list.append(target_table_list)
                            result_list.append(target_table_list)
                            table_count += 1
                            
                            if '[TABLE_END]' not in now_line:
                                table_flag = True # 테이블 시작.
                            else: # 한 line에 '[TABLE_START]'와 '[TABLE_END]' 토큰이 같이 있는 경우
                                table_flag = False
                            
                            past_table_xs = target_table_xs # 현재 테이블의 xs 값들을 이전 테이블의 xs 값들로 저장.
                    else: # 테이블 섹션의 일반 텍스트들은 필요 없으므로 pass.
                        if '[TABLE_END]' in now_line: # 테이블 섹션 끝 위치.
                            table_flag = False # 테이블 끝.

                past_page_bottom = doc_text_list[-1]

        result_list = [self._table_formatting(result) if type(result) == list else result for result in result_list]

        result_text = r'\n'.join(result_list)
        result_text = ' '.join(result_text.split()) # 띄어쓰기가 여러 개 중첩된 경우, 하나로 처리.

        return result_text.replace('[TB_START]', ' ').replace('[TB_END]', ' ')
    
    # 현재 line이 이전 line과 연결될 때, spacing or attatching
    def _check_spacing(self, result_list: list, past_line: str or list, now_line: str) -> str:
        if type(result_list[-1]) == list:
            result_list.append(now_line)
            return ''

        try:
            past_first_word = past_line.split()[0] # 이전 line의 첫 번째 text.
            past_last_word = past_line.split()[-1] # 이전 line의 마지막 text.
        except IndexError: # past_line이 빈 문자열인 경우
            past_first_word = ''
            past_last_word = ''

        now_first_word = now_line.split()[0] # 현재 line의 첫 번째 text.

        # 카테고리 시작 line인 경우
        if re.match('^\d+\.$|^\d+\)$|^\(\d+\)$|^[가-하]\.$|^주\d*\)|^\(주\d*\)', now_first_word) or now_first_word.startswith(self.roma_num) or now_first_word.startswith(self.round_num):
            if past_last_word.endswith(('니', '한', '있', '없', '같', '된', '본', '진', '였', '인', '하', '는', '둔', '했')) and now_first_word.startswith('다.'): # 간혹 "다." 와 같은 텍스트가 카테고리 시작 문자가 아닌, 어떤 문장의 끝을 의미하는 문자인 경우
                result_list[-1] += now_line
                self.sentence_end_flag = True
            else:
                result_list[-1] += r'\n' + now_line # 카테고리 구분 토큰으로 '|' 추가.
            return

        # 카테고리 시작 line 제외한 모든 경우
        else:
            if past_last_word.endswith(('.', ',', '%')) or now_first_word.startswith(self.context_token) or now_first_word.startswith(('및', '그')):
                result_list[-1] += ' ' + now_line
            elif re.match('.*[^\d]$', past_last_word) and re.match('^\d.*', now_first_word):
                result_list[-1] += ' ' + now_line
            elif now_first_word in self.josa or now_first_word in ('.', ',', '%'):
                result_list[-1] += now_line
            elif past_last_word.endswith(self.josa):
                if past_last_word.endswith('으로') and now_first_word.startswith(('서', '써', '는')):
                    result_list[-1] += now_line
                elif past_last_word.endswith('가') and now_first_word.startswith(('액', '설', '능', '정', '된')):
                    result_list[-1] += now_line
                elif past_last_word.endswith('함') and now_first_word.startswith('으로'):
                    result_list[-1] += now_line
                elif past_last_word.endswith('도') and now_first_word.startswith('록'):
                    result_list[-1] += now_line
                elif past_last_word.endswith('에') and now_first_word.startswith(('게', '서')):
                    result_list[-1] += now_line
                elif past_last_word.endswith('해') and now_first_word.startswith('당'):
                    result_list[-1] += now_line
                elif past_last_word.endswith('만') and now_first_word.startswith('져'):
                    result_list[-1] += now_line
                elif past_last_word.endswith('인') and now_first_word.startswith('감'):
                    result_list[-1] += now_line
                else:
                    result_list[-1] += ' ' + now_line
            elif len(past_last_word) == 1:
                if past_last_word == '수':
                    if now_first_word.startswith(('있', '없')):
                        result_list[-1] += ' ' + now_line
                    else:
                        result_list[-1] += now_line
                elif past_last_word == '및':
                    result_list[-1] += ' ' + now_line
                else:
                    result_list[-1] += now_line
            elif re.match('\d.*년$', past_last_word):
                if re.match('^\d.*월', now_first_word):
                    result_list[-1] += ' ' + now_line
                else:
                    result_list[-1] += now_line
            elif re.match('\d.*월$', past_last_word):
                if re.match('^\d.*일', now_first_word):
                    result_list[-1] += ' ' + now_line
                else:
                    result_list[-1] += now_line
            elif re.match('\d.*일$', past_last_word):
                if now_first_word in self.josa:
                    result_list[-1] += now_line
                else:
                    result_list[-1] += ' ' + now_line
            elif re.match('^\d+\.$|^\d+\.\d$|^\d+\)$|^\(\d+\)$|^[가-하]\.$|^주\d*\)|^\(주\d*\)', past_first_word) or past_first_word.startswith(self.roma_num) or past_first_word.startswith(self.round_num): # 카테고리 시작 line의 다음 line인 경우
                if len(re.sub('[A-Za-z]', '', past_line)) < len(re.sub('[A-Za-z]', '', now_line)):
                    result_list[-1] += ' ' + now_line
                elif re.match('.*[가-힣]\)$', past_last_word): # 조금 애매한 로직.
                    result_list[-1] += ' ' + now_line
                else:
                    result_list[-1] += now_line
            else:
                result_list[-1] += now_line
        return

    def _merged_cell_process(self, table_text_list: list) -> list:
        new_table_text_list = []
        # 전처리.
        for row_idx, row_data in enumerate(table_text_list):
            if row_idx != len(table_text_list) - 1: # 테이블 리스트의 마지막 요소는 xs.(dictionary)
                row_data = [str(row_data[cell_idx]).replace('\xa0', '').replace('\n', '') if row_data[cell_idx] != None else None for cell_idx in range(len(row_data))]
            new_table_text_list.append(row_data) # 전처리한 row 텍스트들 하나씩 추가.

        for row_idx, row_data in enumerate(new_table_text_list): # 테이블의 row.
            if type(row_data) != dict: # 테이블 리스트의 마지막 요소(xs)는 제외.
                for cell_idx in range(len(row_data)): # cell 데이터의 index.
                    if row_idx == 0: # key row (첫 번째 row인 경우)
                        key_text = row_data[cell_idx] # 현재 셀의 key text.

                        if key_text == '[COL_MERGED]': # key text가 None인 경우
                            if '[COL_MERGED]' not in new_table_text_list[row_idx][cell_idx - 1]:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx][cell_idx - 1] + '[COL_MERGED]' # 이전 key text와 동일하게 변환.
                            else:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx][cell_idx - 1] # 이미 col_merged 토큰이 있다면, 토큰 추가 x.
                    else: # value row (첫 번째를 제외한 모든 나머지 row인 경우)
                        value_text = row_data[cell_idx] # 현재 셀의 value text.

                        if value_text == '[COL_MERGED]':
                            if '[COL_MERGED]' not in new_table_text_list[row_idx][cell_idx - 1]:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx][cell_idx - 1] + '[COL_MERGED]'
                            else:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx][cell_idx - 1]
                        elif value_text == '[ROW_MERGED]':
                            if '[ROW_MERGED]' not in new_table_text_list[row_idx - 1][cell_idx]:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx - 1][cell_idx] + '[ROW_MERGED]'
                            else:
                                new_table_text_list[row_idx][cell_idx] = new_table_text_list[row_idx - 1][cell_idx]
                            
        return new_table_text_list
    
    def _table_formatting(self, target_table_list: list) -> str:
        if len(target_table_list) > 1:
            try:
                # 테이블의 첫 번째 row가 한 값만 있는 경우
                if not ''.join(target_table_list[0]).replace('[ROW_MERGED]', '').replace('[COL_MERGED]', '').replace(f'{target_table_list[0][0]}', '').strip():
                    if len(target_table_list[0]) == 1: # 테이블의 모든 row에 cell이 하나만 있는 경우
                        formatted_table_str = ''
                        if len(target_table_list) % 2 == 0: # 테이블 row 개수가 짝수인 경우
                            formatted_table_str = ''.join([f'{{{target_table_list[row_idx][0]}|{target_table_list[row_idx + 1][0]}}} ' for row_idx in range(0, len(target_table_list), 2)])
                        else: # 테이블 row 개수가 홀수인 경우
                            formatted_table_str = ''.join([f'{{{row[0]}|' if row_idx == 0 else f' {row[0]}}} ' if row_idx == len(target_table_list) - 1 else f' {row[0]}|' for row_idx, row in enumerate(target_table_list)])
                        return formatted_table_str
                    else: # 테이블의 첫 번째 row만 한 값이 있는 경우 (필요 없음)
                        target_table_list = target_table_list[1:]

                criteria_key_row = ' '.join(target_table_list[1]) # 두 번째 row.
                formatted_table_str = ''
                main_key_range = 1
                if '[ROW_MERGED]' not in criteria_key_row: # 첫 번째 row만 key인 테이블.
                    for row_idx, row in enumerate(target_table_list):
                        if row_idx == 0: # 첫 번째 row.(key)
                            for cell_idx, cell in enumerate(row):
                                if cell_idx == 0:
                                    locals()[f'main_key_{cell_idx}'] = cell # 첫 번째 row의 셀들을 main_key로 설정.
                                else:
                                    if cell.replace('[COL_MERGED]', '') == locals()[f'main_key_0']:
                                        main_key_range += 1 # 첫 번째 key 값이 col_merged인 경우, 그 범위를 저장.

                                    locals()[f'main_key_{cell_idx}'] = cell.replace('[COL_MERGED]', '')
                        else: # 나머지 모든 row.
                            for cell_idx, cell in enumerate(row):
                                cell = cell.replace('[ROW_MERGED]', '').replace('[COL_MERGED]', '')
                                if cell_idx == 0:
                                    sub_key = cell
                                else:
                                    if main_key_range and cell_idx in list(range(main_key_range))[1:]: # col_merged 범위의 sub_key들을 모두 병합.
                                        if cell != sub_key:
                                            sub_key += f' {cell}'
                                    else:
                                        formatted_table_str += f'{{{locals()[f"main_key_0"]} {sub_key} {locals()[f"main_key_{cell_idx}"]}|{cell}}} '
                else: # key row가 row merged인 경우
                    key_row_flag = True
                    for row_idx, row in enumerate(target_table_list):
                        if row_idx == 0: # 첫 번째 row.
                            for cell_idx, cell in enumerate(row):
                                cell = cell.replace('[COL_MERGED]', '')
                                if cell_idx == 0:
                                    locals()[f'main_key_{cell_idx}'] = cell
                                else:
                                    if cell == locals()[f'main_key_0']:
                                        main_key_range += 1 # 첫 번째 key 값이 col_merged인 경우, 그 범위를 저장.

                                    locals()[f'main_key_{cell_idx}'] = cell
                        else: # 나머지 모든 row.
                            if '[ROW_MERGED]' not in ' '.join(row):
                                key_row_flag = False # key row가 끝나는 경우, flag False로 변경.

                            if key_row_flag: # 아직 key row 범위인 경우
                                for cell_idx, cell in enumerate(row):
                                    cell = cell.replace('[ROW_MERGED]', '').replace('[COL_MERGED]', '')
                                    if cell == locals()[f'main_key_{cell_idx}']:
                                        continue
                                    else:
                                        locals()[f'main_key_{cell_idx}'] += f' {cell}'
                            else: # value row.
                                for cell_idx, cell in enumerate(row):
                                    cell = cell.replace('[ROW_MERGED]', '').replace('[COL_MERGED]', '')
                                    if cell_idx == 0:
                                        sub_key = cell
                                    else:
                                        if main_key_range and cell_idx in list(range(main_key_range))[1:]: # col_merged 범위의 sub_key들을 모두 병합.
                                            if cell != sub_key:
                                                sub_key += f' {cell}'
                                        else:
                                            formatted_table_str += f'{{{locals()[f"main_key_0"]} {sub_key} {locals()[f"main_key_{cell_idx}"]}|{cell}}} '
            except:
                formatted_table_str = str(target_table_list)
        else:
            formatted_table_str = str(target_table_list)

        return formatted_table_str.strip()