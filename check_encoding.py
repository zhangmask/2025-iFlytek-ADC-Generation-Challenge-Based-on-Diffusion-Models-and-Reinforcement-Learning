import chardet

# Check encoding of submit1.csv
with open('submit/submit1.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print('submit1.csv encoding:', result['encoding'])

# Check encoding of submit2.csv
with open('submit/submit2.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print('submit2.csv encoding:', result['encoding'])