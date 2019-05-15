import spacy

encoder = spacy.load('en_core_web_md')
txt = "Dan's parents were overweight. Dan was overweight as well. The BBC told his parents it was unhealthy. His parents understood and decided to make a change. They got themselves and Dan on a diet."

txt = encoder(txt)
for sent in txt.sents:
    print(sent)

for word in txt:
    print(word.string, word.tag_, word.pos_)
