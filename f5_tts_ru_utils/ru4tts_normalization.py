"""
Russian text normalization for TTS
https://github.com/shigabeev/russian_tts_normalization/tree/main

Implemented
- Cyrrilization of letters such as "apple" -> "эппл".
- Abbreviations expansion such as "СССР" -> "эс эс эс эр".
- Numbers conversion of any size
- Currency expansion
- Phone number expansion
- Date
- Roman numbers to arabic numbers
"""


import re
import unicodedata
from .patch_dict import word_transliteration_exceptions, abbreviation_exceptions, common_abbreviation_expansions
from functools import lru_cache
from . import config


# Define mapping for digraphs and individual Latin letters
cyrrilization_mapping_extended = {
    # Trigraphs & digraphs from the second dictionary
    'ya': 'я', 'yo': 'ё', 'yu': 'ю', 'zh': 'ж', 'ts': 'ц',

    # Digraphs (non-conflicting from the first dictionary)
    'sh': 'ш', 'ch': 'ч', 'th': 'з', 'ph': 'ф', 'oo': 'у', 'ee': 'и', 'kh': 'х', 'sch': 'ск',

    # Letters (values from the second dictionary take priority)
    'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'х',
    'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
    'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'v': 'в',
    'w': 'в', 'x': 'кс', 'y': 'ы', 'z': 'з'
}



# Russian letter to its phonetic pronunciation mapping
pronunciation_map = {
    'А': 'а', 'Б': 'бэ', 'В': 'вэ', 'Г': 'гэ', 'Д': 'дэ',
    'Е': 'е', 'Ё': 'ё', 'Ж': 'жэ', 'З': 'зэ', 'И': 'и',
    'Й': 'ий', 'К': 'ка', 'Л': 'эл', 'М': 'эм', 'Н': 'эн',
    'О': 'о', 'П': 'пэ', 'Р': 'эр', 'С': 'эс', 'Т': 'тэ',
    'У': 'у', 'Ф': 'эф', 'Х': 'ха', 'Ц': 'цэ', 'Ч': 'чэ',
    'Ш': 'ша', 'Щ': 'ща', 'Ъ': 'твёрдый знак', 'Ы': 'ы', 'Ь': 'мягкий знак',
    'Э': 'э', 'Ю': 'ю', 'Я': 'я'
}

# Roman numeral to integer map
roman_numerals = {
    'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
    'C': 100,  'XC': 90,  'L': 50,  'XL': 40,
    'X': 10,   'IX': 9,   'V': 5,   'IV': 4,
    'I': 1
}
# Strict Roman numeral regex (up to 3999, no single "I")
roman_regex = re.compile(
    r'\bM{0,3}(CM|CD|D?C{0,3})'
    r'(XC|XL|L?X{0,3})'
    r'(IX|IV|V?I{0,3})\b'
)
# Special context-based Roman numeral exception
contextual_exceptions = re.compile(r'\b(Глава|Часть|Раздел|Том)\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\b')

def is_latin_char(ch):
    """Check if a single character is a Latin letter."""
    try:
        return ch.isalpha() and 'LATIN' in unicodedata.name(ch)
    except ValueError:
        return False
        
def roman_to_int(text: str) -> str:
    # First: apply contextual conversion (e.g. Глава I → Глава 1)
    def context_replace(match):
        word = match.group(1)
        roman = match.group(2)
        num = 0
        i = 0
        while i < len(roman):
            if i + 1 < len(roman) and roman[i:i+2] in roman_numerals:
                num += roman_numerals[roman[i:i+2]]
                i += 2
            else:
                num += roman_numerals[roman[i]]
                i += 1
        return f"{word} {num}"

    text = contextual_exceptions.sub(context_replace, text)

    # Then: apply general Roman numeral conversion
    def convert(match):
        roman = match.group(0)
        num = 0
        i = 0
        while i < len(roman):
            if i + 1 < len(roman) and roman[i:i+2] in roman_numerals:
                num += roman_numerals[roman[i:i+2]]
                i += 2
            else:
                num += roman_numerals[roman[i]]
                i += 1
        return str(num)

    return roman_regex.sub(convert, text)


def cyrrilize(text: str) -> str:
    # Protect special tags like [BREAK] using a placeholder map
    protected_tags = re.findall(r'\[BREAK]', text)
    tag_placeholders = {f"⟦{i}⟧": tag for i, tag in enumerate(protected_tags)}
    for placeholder, tag in tag_placeholders.items():
        text = text.replace(tag, placeholder)
    # Tokenize: keep words, whitespace, punctuation
    tokens = re.findall(r'\w+(?:[-’\']\w+)*[.,!?;:]*|\s+|[^\w\s]', text, re.UNICODE)
    result = []

    for token in tokens:
        # Ignore special tags like [BREAK]
        """if re.fullmatch(r'\[BREAK\]', token):
            result.append(token)
            continue
        """    
        if not any(is_latin_char(c) for c in token):
            result.append(token)
            continue

        # Handle full word exceptions (case-insensitive)
        lower_token = token.lower()
        if lower_token in word_transliteration_exceptions:
            translit = word_transliteration_exceptions[lower_token]
            # Preserve capitalization of first letter if original token was capitalized
            if token[0].isupper():
                translit = translit.capitalize()
            result.append(translit)
            continue

        # Default transliteration
        translit = ''
        i = 0
        while i < len(token):
            # Check digraph
            if i + 1 < len(token):
                pair = token[i:i+2].lower()
                if all(is_latin_char(c) for c in pair) and pair in cyrrilization_mapping_extended:
                    mapped = cyrrilization_mapping_extended[pair]
                    translit += mapped
                    i += 2
                    continue
            char = token[i]
            if is_latin_char(char):
                mapped = cyrrilization_mapping_extended.get(char.lower(), char)
                translit += mapped
            else:
                translit += char
            i += 1

        result.append(translit)

    output = ''.join(result)

    # Restore original tags
    for placeholder, tag in tag_placeholders.items():
        output = output.replace(placeholder, tag)

    return output


@lru_cache(maxsize=1000)
def number_to_words(n, gender='masculine'):
    """
    Convert a number into Russian words.
    Supports 'masculine' (default) and 'feminine' forms for 1 and 2.
    """
    if n == 0:
        return 'ноль'

    units_masc = ['','один','два','три','четыре','пять','шесть','семь','восемь','девять']
    units_fem = ['','одна','две','три','четыре','пять','шесть','семь','восемь','девять']
    teens = ['десять','одиннадцать','двенадцать','тринадцать','четырнадцать','пятнадцать','шестнадцать','семнадцать','восемнадцать','девятнадцать']
    tens = ['','десять','двадцать','тридцать','сорок','пятьдесят','шестьдесят','семьдесят','восемьдесят','девяносто']
    hundreds = ['','сто','двести','триста','четыреста','пятьсот','шестьсот','семьсот','восемьсот','девятьсот']

    units = units_masc if gender == 'masculine' else units_fem

    thousand_units = ['тысяча', 'тысячи', 'тысяч']
    million_units = ['миллион', 'миллиона', 'миллионов']
    billion_units = ['миллиард', 'миллиарда', 'миллиардов']

    words = []

    def russian_plural(number, units):
        if number % 10 == 1 and number % 100 != 11:
            return units[0]
        elif 2 <= number % 10 <= 4 and (number % 100 < 10 or number % 100 >= 20):
            return units[1]
        else:
            return units[2]

    def under_thousand(number, gender='masculine'):
        if number == 0:
            return []
        elif number < 10:
            return [units[number]]
        elif number < 20:
            return [teens[number - 10]]
        elif number < 100:
            return [tens[number // 10], units[number % 10]]
        else:
            return [hundreds[number // 100]] + under_thousand(number % 100, gender)

    billions = n // 1_000_000_000
    millions = (n % 1_000_000_000) // 1_000_000
    thousands = (n % 1_000_000) // 1_000
    remainder = n % 1_000

    if billions:
        words += under_thousand(billions) + [russian_plural(billions, billion_units)]
    if millions:
        words += under_thousand(millions) + [russian_plural(millions, million_units)]
    if thousands:
        if thousands % 10 == 1 and thousands % 100 != 11:
            words.append('одна')
        elif thousands % 10 == 2 and thousands % 100 != 12:
            words.append('две')
        else:
            words += under_thousand(thousands, 'feminine')
        words.append(russian_plural(thousands, thousand_units))

    words += under_thousand(remainder, gender)

    return ' '.join(word for word in words if word)



def detect_numbers(text: str) -> str:
    # Regular expression pattern for matching standalone numbers
    number_pattern = re.compile(r'\b\d+\b')
    # Find all matches and return them along with their start and end indices
    matches = list(number_pattern.finditer(text))
    number_matches = [{'number': match.group(), 'start': match.start(), 'end': match.end()} for match in matches]
    
    return number_matches

def number_to_words_digit_by_digit(n):
    """
    Convert a number into its word components in Russian, digit by digit.
    """
    units = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
    return ' '.join(units[int(digit)] for digit in str(n))

# Update the normalize_text_with_numbers to handle large numbers by reading them digit by digit
def normalize_text_with_numbers(text):
    # Detect all standalone numbers in the text
    detected_numbers = detect_numbers(text)
    # Sort detected numbers by their starting index in descending order
    detected_numbers.sort(key=lambda x: x['start'], reverse=True)
    
    # Replace each number with its normalized form
    for num in detected_numbers:
        number_value = int(num['number'])
        # For large numbers that are out of the range of the 'number_to_words' function, use 'number_to_words_digit_by_digit'
        if number_value >= 1_000_000_000_000:
            normalized_number = number_to_words_digit_by_digit(number_value)
        else:
            normalized_number = number_to_words(number_value)
        # Replace the original number in the text with its normalized form
        text = text[:num['start']] + normalized_number + text[num['end']:]
    
    return text


def normalize_phone_number(phone_number):
    # Strip the phone number of all non-numeric characters
    digits = re.sub(r'\D', '', phone_number)

    # Define the segments for the Russian phone number
    segments = {
        'country_code': digits[:1],  # +7 or 8
        'area_code': digits[1:4],    # 495
        'block_1': digits[4:7],      # 123
        'block_2': digits[7:9],      # 45
        'block_3': digits[9:11],     # 67
    }

    # Normalizing the country code
    if segments['country_code'] == '8':
        segments['country_code'] = 'восемь'
    elif segments['country_code'] == '7':
        segments['country_code'] = 'плюс семь'

    # Normalize each segment using the number_to_words function
    normalized_segments = {
        key: number_to_words(int(value)) if key != 'country_code' else value
        for key, value in segments.items()
    }

    # Combine the segments into the final spoken form
    spoken_form = ' '.join(normalized_segments.values())

    return spoken_form

# Correcting the phone number normalization function to handle various formats correctly

def normalize_text_with_phone_numbers(text: str) -> str:
    # Detect all phone numbers in the text 
    """
    🇷🇺 Russian numbers, like:
    +7 (123) 456-78-90
    8 123 456 78 90
    +7 1234567890
    +7(123)4567890
    81234567890

    🇺🇸 US numbers, like:
    123-548-4230
    +1 123-548-4230
    +1 (415) 598-5698
    (415) 598-5698
    """
    phone_pattern = re.compile(
        r"(?:\+7|8)\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}"
        r"|8\d{10}"
        r"|(?:\+1\s?)?(?:\(\d{3}\)|\d{3})[-\s]?\d{3}[-\s]?\d{4}"
    )
    # We use finditer here instead of findall to get the match objects, which will include the start and end indices.
    matches = list(phone_pattern.finditer(text))
    detected_phone_numbers = [{'phone': match.group().strip(), 'start': match.start(), 'end': match.end()} for match in matches]

    # Sort detected phone numbers by their starting index in descending order
    # This ensures that when we replace them, we don't mess up the indices of the remaining phone numbers
    detected_phone_numbers.sort(key=lambda x: x['start'], reverse=True)
    
    # Replace each phone number with its normalized form
    for pn in detected_phone_numbers:
        normalized_phone = normalize_phone_number(pn['phone'])
        # Replace the original phone number in the text with its normalized form
        text = text[:pn['start']] + normalized_phone + text[pn['end']:]
    
    return text

# Full function that detects and converts currency in a text to its full Russian word representation
def currency_normalization(text: str) -> str:
    """
    Detects currency amounts in the text and converts them to their word representations in Russian.
    """
    # Helper function to resolve the correct form of the currency units
    def russian_plural(number, units):
        if number % 10 == 1 and number % 100 != 11:
            return units[0]
        elif 2 <= number % 10 <= 4 and (number % 100 < 10 or number % 100 >= 20):
            return units[1]
        else:
            return units[2]

    # Function to convert a currency amount into its word components in Russian
    def currency_to_words(amount, currency='rub'):
        # Define the currency units and subunits
        currencies = {
            'rub': (['рубль', 'рубля', 'рублей'], ['копейка', 'копейки', 'копеек']),
            'usd': (['доллар', 'доллара', 'долларов'], ['цент', 'цента', 'центов']),
            'eur': (['евро', 'евро', 'евро'], ['евроцент', 'евроцента', 'евроцентов']),  # Euro has invariable form
            'gbp': (['фунт', 'фунта', 'фунтов'], ['пенс', 'пенса', 'пенсов']),
            'uah': (['гривна', 'гривны', 'гривен'], ['копейка', 'копейки', 'копеек']),
        }

        # Get the correct currency units
        main_units, sub_units = currencies.get(currency, currencies['rub'])

        # Separate the amount into main and subunits
        main_amount = int(amount)
        sub_amount = int(round((amount - main_amount) * 100))

        # Convert numbers to words
        main_words = number_to_words(main_amount) + ' ' + russian_plural(main_amount, main_units)
        sub_words = ''

        # Add subunits if present
        if sub_amount > 0:
            sub_words = number_to_words(sub_amount) + ' ' + russian_plural(sub_amount, sub_units)

        # Combine main and subunit words
        full_currency_words = main_words.strip()
        if sub_words:
            full_currency_words += ' ' + sub_words.strip()

        return full_currency_words

    # Define currency patterns for detection
    currency_patterns = {
        'rub': [r'(\d+(?:\.\d\d)?)\s*(руб(л(ей|я|ь))?|₽)', r'(\d+(?:\.\d\d)?)\s*RUB'],
        'usd': [r'(\d+(?:\.\d\d)?)\s*(доллар(ов|а|ы)?|\$)', r'(\d+(?:\.\d\d)?)\s*USD', r'\$(\d+(?:\.\d\d)?)'],
        'eur': [r'(\d+(?:\.\d\d)?)\s*(евро|€)', r'(\d+(?:\.\d\d)?)\s*EUR', r'(\d+)\s*€'],
        'gbp': [r'(\d+(?:\.\d\d)?)\s*(фунт(ов|а|ы)?|£)', r'(\d+(?:\.\d\d)?)\s*GBP', r'£(\d+)'],
        'uah': [r'(\d+(?:\.\d\d)?)\s*(грив(ен|ны|на)|₴)', r'(\d+(?:\.\d\d)?)\s*UAH', r'(\d+)\s*₴'],
    }

    # Detect and convert currencies in the text
    def detect_currency(text: str) -> str:
        # Check each currency pattern to find matches
        for currency_code, patterns in currency_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Extract the amount and convert it to words
                    amount = float(match.group(1))
                    currency_words = currency_to_words(amount, currency_code)
                    # Replace the original amount with its word representation in the text
                    text = re.sub(pattern, currency_words, text, count=1)

        return text

    # Run the detection and conversion on the input text
    return detect_currency(text)

# Updated function to normalize dates in a given text with month names and ordinal days
def normalize_dates(text: str) -> str:
    # Month names in Russian in the genitive case
    month_names = {
        '01': 'января', '02': 'февраля', '03': 'марта',
        '04': 'апреля', '05': 'мая', '06': 'июня',
        '07': 'июля', '08': 'августа', '09': 'сентября',
        '10': 'октября', '11': 'ноября', '12': 'декабря'
    }

    # Regular expression for matching dates in DD.MM.YYYY format
    date_pattern = re.compile(r'\b(\d{2})\.(\d{2})\.(\d{4})\b')

    # Function to normalize a single date
    def normalize_date(match):
        day, month, year = match.groups()
        # Convert day to ordinal word and year to words
        day_word = number_to_words_ordinal(int(day))
        year_word = number_to_words(int(year))
        # Use the month name from the mapping
        month_name = month_names.get(month, '')
        # Construct the normalized date string in the format "7 января 2021 года"
        return f'{day_word} {month_name} {year_word} года'
    
    def number_to_words_ordinal(n):
        """
        Convert a number into its ordinal word components in Russian. This function is specific to days of the month,
        where ordinal numbers are required.
        """
        # Russian ordinal numbers for days (1st to 31st) in the genitive case, which is used for dates
        ordinal_days = {
            1: 'первое', 2: 'второе', 3: 'третье', 4: 'четвёртое', 5: 'пятое',
            6: 'шестое', 7: 'седьмое', 8: 'восьмое', 9: 'девятое', 10: 'десятое',
            11: 'одиннадцатое', 12: 'двенадцатое', 13: 'тринадцатое', 14: 'четырнадцатое', 15: 'пятнадцатое',
            16: 'шестнадцатое', 17: 'семнадцатое', 18: 'восемнадцатое', 19: 'девятнадцатое', 20: 'двадцатое',
            21: 'двадцать первое', 22: 'двадцать второе', 23: 'двадцать третье', 24: 'двадцать четвёртое',
            25: 'двадцать пятое', 26: 'двадцать шестое', 27: 'двадцать седьмое', 28: 'двадцать восьмое',
            29: 'двадцать девятое', 30: 'тридцатое', 31: 'тридцать первое'
        }
        return ordinal_days.get(n, '')

    # Replace all found dates in the text with their normalized forms
    normalized_text = date_pattern.sub(normalize_date, text)

    return normalized_text


def get_hour_suffix(h):
    if h % 10 == 1 and h % 100 != 11:
        return "час"
    elif 2 <= h % 10 <= 4 and (h % 100 < 10 or h % 100 >= 20):
        return "часа"
    else:
        return "часов"

def get_minute_suffix(m):
    if m % 10 == 1 and m % 100 != 11:
        return "минута"
    elif 2 <= m % 10 <= 4 and (m % 100 < 10 or m % 100 >= 20):
        return "минуты"
    else:
        return "минут"

def convert_time_expressions(text: str) -> str:
    def replacer(match):
        hour = int(match.group('hour'))
        minute = int(match.group('minute'))
        ampm = match.group('ampm')

        # Adjust 12-hour clock if AM/PM is used
        if ampm:
            ampm = ampm.lower()
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0

        hour_word = number_to_words(hour)
        hour_suffix = get_hour_suffix(hour)

        if minute == 0:
            minute_word = "ноль ноль"
        elif minute < 10:
            minute_word = f"ноль {number_to_words(minute)}"
        else:
            minute_word = number_to_words(minute, gender='feminine')

        minute_suffix = get_minute_suffix(minute)

        return f"{hour_word} {hour_suffix} {minute_word} {minute_suffix}"

    # This matches 24h or 12h with optional AM/PM
    pattern = re.compile(r'\b(?P<hour>\d{1,2}):(?P<minute>\d{2})(?:\s*(?P<ampm>[APap][Mm]))?\b')
    return pattern.sub(replacer, text)

# Function to expand abbreviations in the text
def process_all_caps_words(text: str, all_caps_to_lower: bool = False) -> str:#, interactive: bool = True) -> str:
    print(f"[DEBUG] all_caps_to_lower received: {all_caps_to_lower}")
    # Now you can use all_caps_to_lower inside this function if needed

    """
    Detects all-caps Russian words and handles them:
    - Expands known abbreviations via `abbreviation_exceptions`
    - If interactive: asks user whether it's an abbreviation
      - If yes: replaces with spelled-out form
      - If no: converts to lowercase
    - If not interactive: default is to convert to lowercase
    """
    context_chars = 30
    # Find all uppercase Cyrillic abbreviations (2+ letters)
    words = re.findall(r'\b[А-ЯЁ]{2,}\b', text)
    processed = set()
    
    for word in words:
        if word in processed:
            continue
        processed.add(word)

        # Rule 1: Known abbreviation → use the exception expansion
        if word in abbreviation_exceptions:
            pronounced_form = abbreviation_exceptions[word]
            text = re.sub(rf'\b{word}\b', pronounced_form, text)
            continue

        if all_caps_to_lower==False:
            # Find context of the word in the text
            start_idx = text.find(word)
            if start_idx == -1:
                # fallback if not found (unlikely)
                context_snippet = word
            else:
                start_context = max(0, start_idx - context_chars)
                end_context = min(len(text), start_idx + len(word) + context_chars)
                context_snippet = text[start_context:end_context]
                # Optionally highlight the word in the context, e.g. uppercase it or surround with []
                # Since word is already uppercase, just show as-is with some marker:
                context_snippet = context_snippet.replace(word, f"[{word}]")

            print(f"\n[DETECTED ALL-CAPS]: {word}")
            print(f"Context: ...{context_snippet}...")
            
            choice = input("Is this an abbreviation? [0/1] (0 is no, 1 is yes): ").strip().lower()
            
            if choice == '0' or choice == '':
                text = re.sub(rf'\b{word}\b', word.lower(), text)
                print("Converted to lower case.")
            elif choice == '1':
                spelled_out = ' '.join(pronunciation_map.get(letter, letter.lower()) for letter in word)
                text = re.sub(rf'\b{word}\b', spelled_out, text)
                print("Spelled out abbreviation.")
            else:
                print("Invalid input. Please enter 0 or 1, or press Enter to keep the current form.")
        else:
            text = re.sub(rf'\b{word}\b', word.lower(), text)

    return text

def expand_abbreviations(text: str) -> str:
    """
    Replaces common abbreviations in the input text with their expanded full forms
    using the common_abbreviation_expansions dictionary.
    """
    for abbr, full in common_abbreviation_expansions.items():
        # Use regex to replace abbreviation as whole word (with optional punctuation)
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text)
    return text



def normalize_russian(text, all_caps_to_lower=False):#, interactive_caps=False):
    text = unicodedata.normalize("NFKC", text) #It might convert symbols like "№" (commonly used in Russian) to a combination of "No".
    text = roman_to_int(text)
    text = process_all_caps_words(text, all_caps_to_lower=all_caps_to_lower)#, interactive=interactive_caps)
    text = expand_abbreviations(text)
    text = normalize_dates(text)
    text = currency_normalization(text)
    text = normalize_text_with_phone_numbers(text)
    text = convert_time_expressions(text)
    text = normalize_text_with_numbers(text)
    text = cyrrilize(text)
    return text
