# =====================================================================================
#  ACORDOR DE CHITARA
#  Flux: microfon -> filtru -> autocorelatie (estimare) -> FFT (precizie) -> stabilizare -> ecran
#  Instalare: pip install numpy scipy pygame sounddevice      Rulare: python acordor.py
#  Iesire: ESC sau X.  Comentariile "de ce" explica alegerea fiecarei valori numerice.
# =====================================================================================
import time
from collections import deque            # lista cu lungime maxima fixa (cade cel mai vechi element)

import numpy as np
import pygame
import sounddevice as sd
from scipy.signal import butter, sosfilt, fftconvolve, find_peaks


# ------------------------------------------------------------------------------------
#  CONSTANTE
# ------------------------------------------------------------------------------------

SR = 44100
# Rata de esantionare. De ce 44100: e rata nativa a aproape oricarui microfon/placa de sunet (fara
# conversii). 22050 ar fi ajuns teoretic (cautam sub 1 kHz), dar nu castigam nimic semnificativ.

WIN = 16384
# Fereastra de analiza: ~0.37 s. De ce: putere a lui 2 (FFT rapid) si cuprinde ~30 perioade din cea mai
# grava nota (82 Hz => perioada 12 ms), deci masuratoarea e stabila. 8192 (0.19 s) dadea valori mai
# tremuratoare pe coardele groase; 32768 (0.74 s) ar face afisarea vizibil intarziata.

NFFT = 4 * WIN
# FFT cu zero-padding 4x: casute de ~0.67 Hz in loc de 2.7 Hz, deci varful se localizeaza mai fin.
# 8x ar costa dublu in timp de calcul fara castig real (precizia o da interpolarea de mai jos).

BIN = SR / NFFT                          # latimea unei casute FFT, in Hz

GATE = 0.0005
# Sub acest volum (abatere standard) tratam semnalul ca liniste. E setat foarte mic intentionat:
# detectia de mai jos (claritate + varf spectral) nu depinde de volum si respinge singura zgomotul,
# deci pragul serveste doar sa nu analizam liniste digitala. La 0.002 pierdeam note slabe (telefon, chitara
# neamplificata); mai jos de ~0.0002 am risca sa analizam doar zgomotul electric al microfonului.

STRINGS = [("E2", 82.41), ("A2", 110.0), ("D3", 146.83), ("G3", 196.0), ("B3", 246.94), ("E4", 329.63)]
# Acordaj standard, temperat egal, A4 = 440 Hz. Distanta minima intre doua coarde alaturate e de 400 centi
# (G3-B3), important pentru histerezisul de selectie a coardei de mai jos.

SOS = butter(4, (60, 1000), "band", fs=SR, output="sos")
# Filtru trece-banda Butterworth (raspuns plat in banda utila, fara "ondulatii").
#  - 60 Hz jos: sub E2 (82 Hz) cu o marja pentru coarde foarte slabite; taie bufniturile. Atenueaza doar
#    ~7 dB la 50 Hz (zumzetul retelei din Europa) - nu pot urca mai mult fara sa afectez E2; zumzetul real
#    e respins mai departe de verificarile de claritate/SNR.
#  - 1000 Hz sus: pastreaza armonica 3 a celei mai inalte note (3 x 330 = 990 Hz), necesara cand fundamentala lipseste.
#  - ordin 4: taiere decenta fara "zbaterea" (ringing) unui ordin mare.
#  - output="sos": forma numerica stabila (varianta cu coeficienti b/a devine instabila la margini joase).

GREEN, ORANGE, RED = (0, 160, 80), (230, 120, 20), (210, 50, 50)
GRAY, LIGHT, BLACK, WHITE = (200, 200, 200), (235, 235, 235), (30, 30, 30), (255, 255, 255)


# ------------------------------------------------------------------------------------
#  FUNCTII AJUTATOARE
# ------------------------------------------------------------------------------------

def cents(f, ref):
    """Diferenta f fata de ref in centi (100 centi = 1 semiton). Scala logaritmica = aceeasi eroare
    perceputa la note joase si inalte, spre deosebire de diferenta in Hz."""
    return 1200 * np.log2(f / ref)


def lerp(a, b, t):
    """Interpolare liniara intre culorile a si b, cu t limitat la [0, 1]."""
    t = max(0, min(1, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# ------------------------------------------------------------------------------------
#  MICROFON
# ------------------------------------------------------------------------------------

buf = np.zeros(WIN, np.float32)          # buffer circular cu ultimii WIN esantioane


def mic(indata, *_):
    """Callback chemat de sounddevice (pe alt fir de executie) la fiecare bloc nou de sunet."""
    global buf
    new = np.roll(buf, -len(indata))     # impingem tot continutul spre stanga cu cat a intrat nou
    new[-len(indata):] = indata[:, 0]    # blocul nou intra la coada (canalul 0 = mono)
    buf = new                            # schimbam referinta dintr-o singura atribuire: fara lacat, citirea din main nu vede un buffer pe jumatate scris


# ------------------------------------------------------------------------------------
#  DETECTIA FRECVENTEI
# ------------------------------------------------------------------------------------

def find_pitch(x):
    """Frecventa fundamentala in Hz, sau None daca sunetul e liniste/zgomot/neclar."""
    x = sosfilt(SOS, x - x.mean())       # scoatem componenta continua, apoi filtram in banda utila
    if x.std() < GATE:                   # volum prea mic: nu analizam
        return None
    x = x * np.hanning(len(x))
    # Fereastra Hann: fara ea, taierea bruta a semnalului produce "scurgeri" spectrale cu lobi laterali care
    # imita varfuri false. Hann are un compromis bun intre latimea varfului si scaderea lobilor laterali.

    # --- Pasul 1: estimare grosiera prin autocorelatie ---
    r = fftconvolve(x, x[::-1])[len(x) - 1:]    # autocorelatia (jumatatea pozitiva), calculata rapid prin FFT
    if r[0] <= 0:                               # energie zero: nimic de analizat
        return None
    r = r / r[0]                                # normalizare: r[0] = 1, deci pragurile de mai jos nu depind de volum

    lo, hi = SR // 400, SR // 70                # intarzieri cautate = frecvente 400..70 Hz
    # De ce 70..400: E2 = 82 Hz si E4 = 330 Hz, cu marja de ~3 semitonuri in jos si ~3.5 in sus ca sa prinda
    # si coardele foarte dezacordate. Un domeniu mai larg doar creste riscul de erori de octava.
    peaks, props = find_peaks(r[lo:hi], height=0.2)
    # height=0.2: ignora ondulatiile mici ale autocorelatiei; sub 0.2 nu poate fi o perioada reala.
    if len(peaks) == 0 or props["peak_heights"].max() < 0.55:
        return None
    # 0.55 = prag de "claritate" (cat de periodic e sunetul). O coarda de chitara da 0.8-0.95; zgomotul
    # alb ramane sub ~0.3. 0.55 lasa marja pentru note slabe/decay fara sa admita zgomot. Mai jos (0.35)
    # a lasat zgomotul sa treaca; mai sus (0.8) ar respinge note reale cu armonici inegale.

    h = props["peak_heights"]
    f0 = SR / (peaks[np.argmax(h >= 0.85 * h.max())] + lo)
    # Luam PRIMUL varf care e >= 85% din maxim. Primul = perioada cea mai scurta = fundamentala; varfurile
    # urmatoare (la 2T, 3T...) sunt aproape la fel de inalte si ar da note cu 1-2 octave mai jos. Pragul de
    # 85% (nu 100%) evita sa sarim peste fundamentala doar fiindca un multiplu iese marginal mai inalt.

    # --- Pasul 2: precizie prin FFT ---
    spec = np.abs(np.fft.rfft(x, NFFT))
    floor = np.median(spec[int(70 / BIN):int(1000 / BIN)]) + 1e-12
    # Nivelul zgomotului = mediana spectrului (nu media: varfurile notei nu o strica). +1e-12 evita /0.

    for m in (1, 2, 3):                  # fundamentala, apoi armonicile 2 si 3 (3 x 330 = 990 < 1000 Hz, limita filtrului)
        # Fallback pe armonici: difuzoarele de telefon nu reda sub ~150 Hz, deci la E2/A2 lipseste fundamentala.
        a, b = int(f0 * m * 0.97 / BIN), int(f0 * m * 1.03 / BIN)
        # Cautam in +-3% in jurul estimarii: eroarea autocorelatiei e ~1% (1 esantion la 400 Hz), deci 3% o
        # acopera cu marja, iar un semiton are ~6%, deci nu putem "sari" pe nota vecina.
        k = a + spec[a:b].argmax()
        sus_destul = 20 * np.log10(spec[k] / floor) >= 15
        # Varful trebuie sa depaseasca zgomotul cu >= 15 dB. Intr-o zona de ~20 casute, varful maxim de zgomot
        # pur iese doar ~6-10 dB peste mediana (masurat: 0 detectii false din 150 de ferestre de zgomot).
        # 20+ dB ar respinge note reale pe microfoane zgomotoase.
        nu_e_artefact = spec[k] > 0.05 * spec[:int(1000 / BIN)].max()
        # Varful trebuie sa fie si >= 5% (-26 dB) din cel mai puternic varf. Fara conditia asta, cand
        # fundamentala lipseste, un lob lateral de la armonici trecea de pragul SNR si dadea o nota gresita.
        if sus_destul and nu_e_artefact:
            l0, l1, l2 = np.log(spec[k - 1:k + 2] + 1e-12)
            d = l0 - 2 * l1 + l2
            return (k + (0.5 * (l0 - l2) / d if d else 0)) * BIN / m
            # Interpolare parabolica pe logaritm (un varf Hann in log e aproape o parabola): pozitia reala a
            # varfului intre casute => precizie de fractiuni de cent. "/ m" revine de la armonica la fundamentala.
    return None


# ------------------------------------------------------------------------------------
#  STABILIZARE
# ------------------------------------------------------------------------------------

class Smooth:
    """Netezeste frecventa si respinge bruiajele. "s" = instanta (echivalentul lui self)."""

    def __init__(s):
        s.f = None                       # frecventa stabilizata curenta
        s.t = 0                          # momentul ultimei masuratori acceptate
        s.hist = deque(maxlen=7)         # ultimele masuratori acceptate
        s.cand = []                      # candidati consecutivi pentru o nota noua

    def update(s, p, now):
        """p = frecventa masurata acum (sau None). Intoarce frecventa stabilizata (sau None)."""
        if p is None:
            s.cand = []                  # confirmarea cere masuratori CONSECUTIVE, deci o intrerupere le anuleaza
            if now - s.t > 1.0:          # 1 s de "hold": ai timp sa citesti valoarea dupa ce nota se stinge
                s.f = None
            return s.f

        if s.f and abs(cents(p, s.f)) < 60:
            # Prag 60 centi: peste jitterul normal (cativa centi) si peste vibrato, sub un semiton (100).
            # O coarda trasa cu cheia se misca continuu, deci ramane in acest interval pas cu pas si e urmarita.
            s.cand = []
            s.t = now
            s.hist.append(p)
            s.f *= 2 ** (0.2 * cents(np.median(s.hist), s.f) / 1200)
            # Mediana pe 7 valori (~0.4 s): un singur numar aberant nu o influenteaza (ar fi nevoie de 4 aberante).
            # Facem doar 20% din drum spre ea: constanta de timp ~5 analize (~0.3 s). 0.5 tremura vizibil,
            # 0.05 ar lasa afisarea cu >1 s in urma.
        else:
            s.cand = (s.cand + [p])[-4:]
            # O masuratoare departe de nota curenta e fie zgomot, fie o nota noua. Nu o credem pe loc: cerem 4
            # consecutive (~0.24 s) si apropiate intre ele (<35 centi). 2 ar lasa sa treaca doua bruiaje la
            # rand; 8 ar face schimbarea corzii sa para lenta.
            if len(s.cand) == 4 and cents(max(s.cand), min(s.cand)) < 35:
                s.f = float(np.median(s.cand))
                s.t = now
                s.hist = deque(s.cand, maxlen=7)
                s.cand = []
        return s.f


# ------------------------------------------------------------------------------------
#  PROGRAM PRINCIPAL
# ------------------------------------------------------------------------------------

def main():
    pygame.init()
    scr = pygame.display.set_mode((640, 420))
    pygame.display.set_caption("Acordor chitara")
    big = pygame.font.SysFont("arial", 96, bold=True)    # numele notei
    mid = pygame.font.SysFont("arial", 30)               # Hz si mesaj
    small = pygame.font.SysFont("arial", 20)             # etichete

    def text(t, font, col, pos):
        """Deseneaza textul t centrat in pos."""
        s = font.render(t, True, col)
        scr.blit(s, s.get_rect(center=pos))

    stream = sd.InputStream(samplerate=SR, channels=1, blocksize=1024, callback=mic)
    # blocksize=1024 (23 ms): compromis intre latenta si numarul de apeluri la callback. Streamul trebuie
    # pastrat intr-o variabila: daca obiectul e colectat de garbage collector, microfonul se inchide.
    stream.start()

    sm = Smooth()
    clock = pygame.time.Clock()
    last = 0                             # momentul ultimei analize
    f, idx, c, tuned, needle = None, None, 0.0, False, 0.0
    # f = frecventa stabilizata; idx = coarda tinta; c = abatere in centi; tuned = acordat?; needle = pozitia (animata) a acului

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        # --- Analiza ---
        now = time.time()
        if now - last > 0.06:
            # ~17 analize/s. Mai des n-ar aduce informatie noua (ferestrele de 0.37 s se suprapun aproape complet);
            # costul unei analize e de ordinul milisecundelor, deci 0.06 s lasa procesorul aproape liber.
            last = now
            f = sm.update(find_pitch(buf.copy()), now)       # copy(): lucram pe un instantaneu, nu pe buffer-ul care se modifica

            if f is None:
                idx, tuned = None, False
            else:
                best = min(range(6), key=lambda i: abs(cents(f, STRINGS[i][1])))   # coarda cu abaterea cea mai mica
                if idx is None or abs(cents(f, STRINGS[best][1])) + 40 < abs(cents(f, STRINGS[idx][1])):
                    idx = best
                # Histerezis de selectie: schimbam coarda tinta doar daca alta e mai aproape cu >= 40 centi.
                # Coardele sunt la >= 400 centi una de alta, deci 40 nu intarzie o schimbare reala, dar opreste
                # saltul intre doua coarde cand nota e la jumatatea distantei.
                c = cents(f, STRINGS[idx][1])
                tuned = abs(c) <= (9 if tuned else 5)
                # Histerezis "acordat": intri sub 5 centi (pragul uzual la acordoarele de chitara; urechea abia
                # sesizeaza ~5 centi), dar iesi abia peste 9, ca textul sa nu palpaie cand esti fix la limita.

        # --- Culoare: verde cand acordat, altfel gradient portocaliu -> rosu ---
        if f is None:
            col = GRAY
        elif tuned:
            col = GREEN
        else:
            col = lerp(ORANGE, RED, (abs(c) - 9) / 30)
            # Gradient continuu (nu praguri) ca sa nu apara salturi de culoare; incepe la 9 centi (iesirea din
            # "acordat") si ajunge la rosu complet la 39 centi.

        # --- Desen ---
        scr.fill(WHITE)
        if f is not None:
            text(STRINGS[idx][0], big, col, (320, 80))
            text(f"{f:.1f} Hz   (tinta {STRINGS[idx][1]:.1f} Hz)", mid, BLACK, (320, 150))
            if tuned:
                msg = "ACORDAT"
            elif c < 0:
                msg = "Prea jos - strange coarda"
            else:
                msg = "Prea sus - slabeste coarda"
            text(msg, mid, col, (320, 330))
        else:
            text("--", big, GRAY, (320, 80))
            text("Canta o coarda...", mid, GRAY, (320, 150))

        target = max(-50, min(50, c)) if f is not None else 0.0
        # Limitam la +-50 centi (jumatate de semiton): dincolo de asta nota e deja mai aproape de alt semiton,
        # iar bara ar iesi din ecran.
        needle += (target - needle) * 0.15
        # Animatie: 15% din distanta ramasa pe cadru (60 fps) => ~0.2 s ca sa ajunga la tinta. Mai mare = acul
        # urmareste prea nervos orice variatie; mai mic = pare blocat.

        pygame.draw.rect(scr, LIGHT, (70, 223, 500, 24), border_radius=12)                  # bara: 500 px = 100 centi => 5 px/cent
        pygame.draw.rect(scr, (200, 235, 215), (320 - 25, 223, 50, 24), border_radius=6)    # zona "acordat": +-5 centi = +-25 px
        for v in (-50, -25, 0, 25, 50):                                                     # gradatii
            x = 320 + v * 5
            pygame.draw.line(scr, GRAY, (x, 253), (x, 265), 2)
        text("-50", small, GRAY, (70, 283))
        text("0", small, GRAY, (320, 283))
        text("+50", small, GRAY, (570, 283))
        pygame.draw.circle(scr, col, (int(320 + needle * 5), 235), 14)                      # acul

        for i in range(6):                                   # cele 6 coarde; cea auzita e colorata
            x = 70 + i * 100
            if f is not None and i == idx:
                pygame.draw.circle(scr, col, (x, 385), 24)
                text(STRINGS[i][0], small, WHITE, (x, 385))
            else:
                pygame.draw.circle(scr, LIGHT, (x, 385), 24)
                text(STRINGS[i][0], small, BLACK, (x, 385))

        pygame.display.flip()
        clock.tick(60)                                       # plafon de 60 fps (suficient de lin, nu consuma procesor degeaba)

    stream.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
