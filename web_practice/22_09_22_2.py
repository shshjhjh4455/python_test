from selenium.webdriver.common.keys import Keys

location ="http://www.yes24.com/Main/default.aspx"
SEARCH = "파이썬"
TIMEOUT = 3
  
try:
    service = Service(executable_path="./chromedriver")
    driver = webdriver.Chrome(service=service)    #브라우저 실행시키위한 드라이버 객체 생성 (브라우저 실행)
    driver.get(location)                          #url 요청
    driver.implicitly_wait(TIMEOUT)               #응답데이터 받을때까지 지연

    search = driver.find_element(by=By.ID, value="query")  #응답된 웹페이지에서 ID가 query인 문서 객체(Element)를 추출
    search.clear()                      #검색어 입력란 clear처리
    search.send_keys(SEARCH)             #검색어 입력
    search.send_keys(Keys.RETURN)      #엔터키 입력 , 검색어가 서버에 전송됨
   
    #검색결과 응답 페이지로부터 검색 결과 책 제목의 요소들 추출
    elements = driver.find_elements(by=By.CSS_SELECTOR, value="div.item_info > div.info_row.info_name > a.gd_name")
    print(len(elements))                             #추출된 요소 개수 출력
    driver.implicitly_wait(TIMEOUT)

    for idx, title in enumerate(elements):    #추출된 요소중 10개의 책 제목만 출력
        if idx==10 : break
        print(title.text)

except Exception:
        raise
finally:
     if driver is not None:
        driver.quit()