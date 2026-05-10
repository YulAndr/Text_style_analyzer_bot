import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BOT_CONFIG = {
    'intents': {
        'writing_style': {
            'examples': ['hottest', 'scruffi', 'onli', 'squar', 'love', 'hasn',  'thought', 'know', 'ban', 'ask', 
                         'vernon', 'figg', 'happen', 'tempt', 'lawn', 'breez', 'vandalis', 'away', 'emerald', 'sorri', 
                         'children', 'boy', 'due', 'stupid', 'window', 'well', 'whole', 'earth', 'slightli', 'look', 
                         'scaveng', 'stone', 'televis', 'drowsi', 'car', 'harri', 'depriv', 'draw', 'prime', 'tri', 
                         'spoke', 'vanish', 'summer', 'back', 'petunia', 'voic', 'smoke', 'dim', 'round', 'outdoor', 
                         'yellow', 'fell', 'law', 'drought', 'flat', 'suppress', 'walk', 'number', 'teeth', 'shade', 
                         'even', 'polkiss', 'night', 'watch', 'mutter', 'whing', 'shh', 'thrown', 'dusti', 'grown', 
                         'ambl', 'popular', 'perfectli', 'peopl', 'conceal', 'invis', 'gleam', 'hair', 'doubt', 'wide', 
                         'silent', 'dursley', 'appear', 'open', 'straight', 'short', 'unconcern', 'green', 'grind', 
                         'pinch', 'seen', 'breakfast', 'dudder', 'lie', 'slowli', 'snort', 'retreat', 'hear', 'past', 
                         'space', 'holiday', 'glare', 'teenag', 'wisteria', 'room', 'recent', 'jingl', 'idea', 'bin', 
                         'jean', 'play', 'parch', 'hot', 'privet', 'old', 'taken', 'nonexist', 'fade', 'bespectacl', 
                         'drive', 'dirti', 'scathingli', 'close', 'clue', 'go', 'pass', 'anyway', 'silenc', 'minist', 
                         'sit', 'person', 'dear', 'suddenli', 'hidden', 'baggi', 'news', 'wander', 'question', 'view'
                         'loudli', 'stuck', 'newspap', 'sort', 'uncl', 'mow', 'street', 'flowerb', 'cereal', 'normal', 
                         'grunt', 'wash', 'aunt', 'larg', 'swallow', 'nearbi', 'wit', 'park', 'member', 'potter', 'sole', 
                         'listen', 'live', 'peel', 'float', 'littl', 'butt', 'upper', 'skinni', 'spot', 'corner', 'stood', 
                         'congratul', 'inhabit', 'dudley', 'shirt', 'difficulti', 'black', 'left', 'throw', 'trainer', 
                         'comfort', 'frown', 'gang', 'spent', 'bush', 'torn', 'knew', 'bran', 'nasti', 'hydrangea', 
                         'flutter', 'care', 'astonishingli', 'fondli', 'batti', 'friend', 'punish', 'hide', 'son', 
                         'fruit', 'neighbour', 'mr', 'hosepip', 'pursuit', 'passer', 'head', 'glad', 'unhealthi', 
                         'shoot', 'ladi', 'endear', 'hope', 'cool'],
            'responses': ['This text style is writing style!', 'I think this is writing style!', 'Writing style, haha!']
        },
        'news_style': {
            'examples': ['abus', 'power', 'laura', 'love', 'truth', 'spark', 'know', 'ask', 'address', 'career', 'daytim', 
                         'brush', 'kevin', 'scene', 'chief', 'boss', 'happen', 'secretari', 'independ', 'complaint', 
                         'took', 'accus', 'imbal', 'award', 'sourc', 'boy', 'revel', 'answer', 'join', 'transpar', 'morn', 
                         'vow', 'mistak', 'televis', 'fallout', 'lover', 'process', 'mail', 'rais', 'dame', 'lygo', 
                         'kuenssberg', 'gordon', 'investig', 'firm', 'nadin', 'senior', 'year', 'sky', 'regular', 'editor', 
                         'carpet', 'toxic', 'show', 'unwis', 'affair', 'level', 'colleagu', 'alleg', 'engulf', 'mccall', 
                         'schofield', 'contact', 'product', 'sought', 'widen', 'friday', 'carolyn', 'singh', 'peopl', 
                         'call', 'pace', 'discrimin', 'job', 'ranj', 'gossip', 'certainli', 'tonight', 'network', 'imper', 
                         'ad', 'insist', 'just', 'sunday', 'felt', 'apolog', 'previous', 'lie', 'think', 'account',
                         'bbc', 'first', 'place', 'conduct', 'meant', 'yesterday', 'drop', 'dorri', 'taken', 'illeg', 
                         'ceremoni', 'rumour', 'controversi', 'tv', 'scorer', 'involv', 'say', 'admit', 'gather', 
                         'channel', 'quizz', 'specialist', 'bulli', 'reveal', 'refus', 'grow', 'person', 'made', 'make', 
                         'detail', 'period', 'romanc', 'news', 'potenti', 'question', 'slater', 'situat', 'emma', 
                         'safeguard', 'phillip', 'gormley', 'move', 'relationship', 'explain', 'lodg', 'told', 'crisi', 
                         'actual', 'held', 'increas', 'age', 'assist', 'shunt', 'men', 'claim', 'insid', 'employe', 
                         'known', 'work', 'least', 'stood', 'director', 'daili', 'prompt', 'pressur', 'young', 'left', 
                         'overseen', 'doctor', 'weekend', 'point', 'frizel', 'programm', 'knew', 'subject', 'lawyer', 
                         'concern', 'spokesman', 'screen', 'cultur', 'tatter', 'awar', 'manag', 'need', 'itv', 'deepen', 
                         'mr', 'comment', 'martin', 'execut', 'head', 'richard', 'won', 'younger', 'scandal', 'inquiri', 
                         'host', 'compani', 'guest', 'parti'],
            'responses': ['This text style is news style!', 'I think this is news style!', 'News style, haha!']
        },
        'scientific_writing': {
            'examples': ['onli', 'english', 'mean', 'grammar', 'compris', 'know', 'continuum', 'soup', 'interact', 
                         'typolog', 'root', 'convent', 'formal', 'read', 'problemat', 'construct', 'array', 'deriv', 
                         'answer', 'given', 'acquisit', 'fillmor', 'pragmat', 'ubiquit', 'instanc', 'colloc', 'vari', 
                         'gener', 'syntact', 'mail', 'rule', 'knowledg', 'perspect', 'book', 'chang', 'consist', 
                         'regular', 'part', 'ditransit', 'rich', 'directli', 'properti', 'participl', 'constitut', 'fli', 
                         'phonolog', 'phrase', 'second', 'factor', 'call', 'cognit', 'central', 'conclud', 'embrac', 
                         'specif', 'use', 'addit', 'structur', 'open', 'occur', 'just', 'sign', 'rang', 'phenomena', 
                         'object', 'peculiar', 'notori', 'schemat', 'slot', 'influenc', 'goldberg', 'idiosyncrat', 'idea', 
                         'highli', 'exhibit', 'deepli', 'verb', 'follow', 'transfer', 'present', 'regularli', 'assum', 
                         'idiom', 'kay', 'view', 'state', 'proven', 'broad', 'develop', 'normal', 'entir', 'kind', 
                         'languag', 'larg', 'linguist', 'extrem', 'contain', 'altern', 'degre', 'least', 'aspect', 'allow', 
                         'behav', 'express', 'negat', 'pair', 'last', 'respect', 'area', 'descript', 'associ', 'semant', 
                         'lexicon', 'noun', 'consider', 'progress', 'except', 'need', 'necessarili', 'emerg', 'exhaust', 
                         'appli', 'inventori', 'combin', 'posit', 'form', 'composit'],
            'responses': ['This text style is scientific writing!', 'I think this is scientific writing!', 
                          'Scientific writing, haha!']
        },
        'officialese': {
            'examples': ['abus''onli', 'english', 'power', 'repeatedli', 'usurp', 'elect', 'system', 'truth', 'swarm', 
                         'amount', 'governor', 'salari', 'uncomfort', 'life', 'unanim', 'wage', 'captiv', 'independ', 
                         'lay', 'represent', 'away', 'district', 'abdic', 'obtain', 'suffer', 'measur', 'deriv',
                         'dissolut', 'press', 'accommod', 'unworthi', 'superior', 'shewn', 'migrat', 'neglect', 'high', 
                         'principl', 'total', 'administr', 'fellow', 'repres', 'mankind', 'condit', 'requir', 'assent', 
                         'earth', 'foundat', 'mercenari', 'quarter', 'guard', 'unit', 'legisl', 'exampl', 'give', 'hand', 
                         'offenc', 'depend', 'peac', 'scarc', 'rule', 'depriv', 'rais', 'render', 'tri', 'kept',
                         'arbitrari', 'trade', 'death', 'organ', 'coloni', 'inestim', 'chang', 'firm', 'effect', 
                         'wholesom', 'endow', 'danger', 'declar', 'payment', 'part', 'benefit', 'law', 'duti', 'entitl', 
                         'substanc', 'equal', 'futur', 'prove', 'fatigu', 'new', 'expos', 'tax', 'purpos', 'pursu', 
                         'safeti', 'unusu', 'constitut', 'militari', 'trial', 'prudenc', 'countri', 'decent', 'encourag', 
                         'polit', 'peopl', 'govern', 'mock', 'call', 'despot', 'affect', 'town', 'brethren', 'plunder', 
                         'will', 'tyrant', 'separ', 'enlarg', 'utterli', 'prevent', 'exercis', 'valuabl', 'repeat', 
                         'jurisdict', 'burnt', 'take', 'just', 'formid', 'invari', 'end', 'invas', 'alter', 'hold', 
                         'unacknowledg', 'experi', 'endeavour', 'fit', 'human', 'fundament', 'object', 'unalien',
                         'commit', 'introduc', 'bear', 'troop', 'perfidi', 'impos', 'place', 'obstruct', 'establish', 
                         'submit', 'forbidden', 'patient', 'opinion', 'record', 'ravag', 'taken', 'destruct', 'erect', 
                         'necess', 'justic', 'evid', 'distant', 'king', 'suspend', 'secur', 'legislatur', 'pass', 
                         'tyranni', 'juri', 'present', 'attend', 'creator', 'assum', 'refus', 'great', 'execution', 
                         'instrument', 'popul', 'coast', 'made', 'band', 'liberti', 'incap', 'barbar', 'invest', 
                         'constrain', 'oper', 'depositori', 'injuri', 'connect', 'arm', 'destroy', 'appropri', 'alon', 
                         'judg', 'state', 'happi', 'annihil', 'institut', 'multitud', 'cours', 'god', 'transient', 'caus', 
                         'compleat', 'larg', 'oppos', 'absolut', 'complianc', 'light', 'impel', 'like', 'case', 'accustom', 
                         'sole', 'fall', 'harrass', 'age', 'live', 'histori', 'public', 'circumst', 'abolish', 'men', 
                         'land', 'armi', 'transport', 'evinc', 'charter', 'relinquish', 'reduc', 'pretend', 'evil', 
                         'boundari', 'work', 'dispos', 'civil', 'consent', 'natur', 'inhabit', 'right', 'creat', 'train', 
                         'world', 'direct', 'thirteen', 'begun', 'throw', 'murder', 'event', 'free', 'dictat', 'design', 
                         'remain', 'cut', 'candid', 'judiciari', 'subject', 'protect', 'import', 'desol', 'foreign', 
                         'friend', 'return', 'punish', 'provinc', 'citizen', 'britain', 'station', 'let', 'parallel', 
                         'cruelti', 'convuls', 'neighbour', 'america', 'combin', 'pursuit', 'sea', 'head', 'tenur', 
                         'dissolv', 'manli', 'offic', 'nation', 'form'],
            'responses': ['This text style is officialese!', 'I think this is officialese!', 'Officialese, haha!']
        }
    }
}

def clean(text): #функция очистки по регистру
    clean_text = ''
    for ch in text.lower():
        if ch in 'abcdefghijklmnopqrstuvwxyz ':
            clean_text = clean_text + ch
    return clean_text
            
def get_intent(text): #функция сравнения
    for intent in BOT_CONFIG ['intents'].keys():
        for example in BOT_CONFIG ['intents'][intent]['examples']:
            s1 = clean(text)
            s2 = clean(example)
            if nltk.edit_distance(s1, s2) / max(len(s1), len(s2)) < 0.4:
                return intent
    return 'intent not defined'
            
def bot(input_text): #функция рандомный ответов
    intent = get_intent(input_text)
    if intent != 'intent not defined':
        return random.choice(BOT_CONFIG ['intents'][intent]['responses'])
    else:
        return 'intent not defined'

#Обучение модели
X = []
y = []
for intent in BOT_CONFIG ['intents'].keys():
        for example in BOT_CONFIG ['intents'][intent]['examples']:
            X.append(example)
            y.append(intent)
len(X), len(y), len(set(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 1)
len(X_train), len(X_test)

vectorizer = CountVectorizer(analyzer = 'word', ngram_range=(1, 3))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
len(vectorizer.get_feature_names())

log_reg = LogisticRegression(C = 0.2)
log_reg.fit(X_train_vect, y_train)
log_reg.score(X_test_vect, y_test)

def get_intent_by_model(text):
    return log_reg.predict(vectorizer.transform([text]))[0]
get_intent_by_model('intents')

def bot(input_text): #функция рандомный ответов
    intent = get_intent_by_model(input_text)
    return random.choice(BOT_CONFIG ['intents'][intent]['responses'])

input_text = '' #печатает ответ
while input_text != 'stop':
    input_text = input()
    if input_text != 'stop':
        response = bot(input_text)
        print(response)