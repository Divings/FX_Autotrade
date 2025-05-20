import matplotlib.font_manager as fm

# 利用可能なフォント名を表示
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    try:
        font_name = fm.FontProperties(fname=font).get_name()
        print(font_name)
    except Exception:
        pass
input(" >> ")