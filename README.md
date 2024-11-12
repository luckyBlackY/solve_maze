プログラミングの勉強を始める時によく出てくるものの一つとして **迷路** がある。迷路の解き方は色々存在する。

- 深さ優先探索
    - 行き止まりや訪問済みの場所に到達したら、直前の分岐に戻る。未探索ルートを探していき、ゴールを目指す。
- 幅優先探索
    - スタートから隣接するマスに進み、キューに入れる。そのキューから順に探索を行って、再びキューに追加していく。
- 壁当てアルゴリズム
    - 壁に沿って右の壁を伝ってゴールに到達するまで進む。
- 強化学習
    - 迷路を解くエージェントが報酬最大化するように環境と相互作用しながら学習していく。十分に学習を行うことで、スタートからゴールへの行き方を学べる。

他にも、A*アルゴリズムやダイクストラ法などがある(......らしい)。

ところで、ここで簡単な迷路を **『直感で』** 解いて頂きたい。その時、なるべく早く正解ルートを導くことを意識して頂きたい(念のために言うが、正解ルートを瞬殺で導く裏の方法があると言うわけでない。どれくらい早く正解ルートを直感的に頭に思い浮かべることができるかを知りたいだけである)。


こちらが、正解ルートを導いてほしい迷路である。左上がスタート, 右下がゴールである。

<details>
    <summary>ここをクリックして迷路を表示</summary>
    (迷路盤面.png)
</details>

<br>

 いかがだろうか。何秒くらいで解けただろうか。1秒？2秒？ 少なくとも、かなり時間を使うということはなかっただろう。

どれくらいかかったかは置いといて、何となく正解ルートが見えたのではないだろうか。

#### なぜ、正解ルートが見えるのか？

おそらく、これまでの **経験** と **画像の雰囲気** からなんとなーく正解ルートが見えたのではないかと思う。

それでふと思ったこと。<br>
#### AIモデルに迷路画像を大量に入力し(経験)、画像の特徴(雰囲気)を学ばせることができたら、迷路が解けるのではないか？

ということで、自己学習のテーマが決まった。
### テーマ：迷路画像を入力し、正解ルート描写迷路画像を出力させる。
後ほど詳細を書くが、学習時はAIモデルに迷路画像と正解ルート描写迷路画像を入れる。それによって、新規の迷路画像で正解ルート描写画像を出力させる。

ここで重要なこととしては、 **迷路のルールは一切教えていない。** つまり、スタートからゴールまでの道を塗ることは教えず、さらにそもそもスタート, ゴール, 壁の概念すら教えず、AI自らにスタートマスとゴールマスを壁を超えずに繋ぐように学習してもらう。

### これが出来ると嬉しいこと：
上記で紹介した方法で迷路を解く時、迷路画像を使用しているわけではなく、プログラム上で迷路の盤面を **再現** している。もし解きたい迷路があった場合、その迷路をプログラム上に再現する手間が発生。<br>
それに対して、今回の方法では **迷路画像をそのまま使用可能。** さらに、元画像に正解ルートを描写してくれるため、正解がわかりやすい。上記の方法よりも手間がかなり省ける。

よって、これが成功したらかなり面白いと思ったため、このテーマで自己学習を行うことに決定した。

## データセット作成
以下のコードによって、マス目の個数, マス目の大きさ, 画像の大きさ, 画像の枚数を指定して、迷路を作成する。作成方法は深さ優先探索。今回は以下の数を指定して、データセットを作成した。
- マス目の個数 ... 9 (盤面は81マス)
- マス目の大きさ ... 14 (画像からはみ出ない、かつ画像全体に迷路が映るよう調整)
- 画像の大きさ ... 128 (学習する時に画像一辺が2^nである必要があるため)
- 画像の枚数 ... 5000 (4000件は学習用、1000件は検証用)

画像例：

| 迷路画像 | 正解ルート描写画像 |
|---|---|
| (迷路盤面.png) | (迷路正解.png) |

<details>
<summary>データセット作成コード</summary>

```python
import random
import numpy as np
from PIL import Image
import os

# 迷路生成のパラメータ
maze_size = (9, 9)    # 迷路の盤面サイズ
cell_size = 14        # 各セルのサイズ（ピクセル）
num_mazes = 5000        # 生成する迷路の数
output_dir_maze = "mazes/maze"  # 迷路画像を保存するディレクトリ
output_dir_solution = "mazes/solution"  # 正解ルート付き迷路画像を保存するディレクトリ

# 画像全体のサイズとパディングの計算
image_size = 128  # 画像全体のサイズ（ピクセル）
maze_pixels = maze_size[0] * cell_size  # 迷路部分のサイズ（ピクセル）
padding = (image_size - maze_pixels) // 2  # 上下左右のパディング（ピクセル）

# 迷路生成のための方向
DIRECTIONS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

# 壁がある状態で迷路のグリッドを初期化
def init_maze(size):
    return np.zeros(size, dtype=int)

# 深さ優先探索によって、1つの解だけを持つ迷路を生成
def carve_passages_from(x, y, grid):
    directions = list(DIRECTIONS.keys())
    random.shuffle(directions)
    for direction in directions:
        nx, ny = x + DIRECTIONS[direction][0], y + DIRECTIONS[direction][1]
        if 0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and grid[nx, ny] == 0:
            grid[x, y] |= (1 << list(DIRECTIONS.keys()).index(direction))  # 現在のセルをマーク
            grid[nx, ny] |= (1 << list(DIRECTIONS.keys()).index(OPPOSITE[direction]))  # 次のセルをマーク
            carve_passages_from(nx, ny, grid)

# 迷路を描画して、グレースケール画像として保存
def draw_maze(grid, maze_number, solution_path=None, is_solution=False):
    # 白（255）で背景画像を初期化
    maze_image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # スタートとゴールのセル（灰色 = 192）
    start_x = padding
    start_y = padding
    goal_x = padding + (maze_size[0] - 1) * cell_size
    goal_y = padding + (maze_size[1] - 1) * cell_size
    maze_image[start_x:start_x + cell_size, start_y:start_y + cell_size] = 192  # スタートのセル
    maze_image[goal_x:goal_x + cell_size, goal_y:goal_y + cell_size] = 192  # ゴールのセル

    # 解答ルートが提供された場合、それを描画（薄い灰色 = 224）
    if solution_path:
        for (x, y) in solution_path:
            if (x, y) != (0, 0) and (x, y) != (maze_size[0] - 1, maze_size[1] - 1):  # スタートとゴールを除外
                x_pixel = padding + x * cell_size
                y_pixel = padding + y * cell_size
                maze_image[x_pixel:x_pixel + cell_size, y_pixel:y_pixel + cell_size] = 224

    # 壁（黒 = 0）を描画 - 壁は最後に描画され、解答ルートの上に表示される
    for x in range(maze_size[0]):
        for y in range(maze_size[1]):
            x_pixel = padding + x * cell_size
            y_pixel = padding + y * cell_size
            if grid[x, y] & (1 << 0) == 0:  # 北の壁
                maze_image[x_pixel, y_pixel:y_pixel + cell_size] = 0
            if grid[x, y] & (1 << 1) == 0:  # 南の壁
                maze_image[x_pixel + cell_size - 1, y_pixel:y_pixel + cell_size] = 0
            if grid[x, y] & (1 << 2) == 0:  # 東の壁
                maze_image[x_pixel:x_pixel + cell_size, y_pixel + cell_size - 1] = 0
            if grid[x, y] & (1 << 3) == 0:  # 西の壁
                maze_image[x_pixel:x_pixel + cell_size, y_pixel] = 0

    # 迷路画像を適切なディレクトリに保存
    if not os.path.exists(output_dir_maze):
        os.makedirs(output_dir_maze)
    if not os.path.exists(output_dir_solution):
        os.makedirs(output_dir_solution)
    
    if is_solution:
        maze_filename = os.path.join(output_dir_solution, f"solution_{maze_number}.png")
    else:
        maze_filename = os.path.join(output_dir_maze, f"maze_{maze_number}.png")
    
    # 画像を保存
    Image.fromarray(maze_image).save(maze_filename)

# スタート (0, 0) からゴール（右下）への解答ルートを見つける
def find_solution_path(grid, start, goal):
    stack = [start]
    visited = set()
    parent = {}
    while stack:
        cell = stack.pop()
        if cell == goal:
            path = []
            while cell != start:
                path.append(cell)
                cell = parent[cell]
            path.append(start)
            return path[::-1]
        visited.add(cell)
        x, y = cell
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and (nx, ny) not in visited:
                if grid[x, y] & (1 << list(DIRECTIONS.keys()).index(direction)):
                    stack.append((nx, ny))
                    parent[(nx, ny)] = (x, y)
    return []

# 指定された数の迷路を生成し、それらを保存
for i in range(num_mazes):
    maze_grid = init_maze(maze_size)
    carve_passages_from(0, 0, maze_grid)  # 左上 (0, 0) から迷路を掘り始める
    solution_path = find_solution_path(maze_grid, (0, 0), (maze_size[0] - 1, maze_size[1] - 1))
    draw_maze(maze_grid, i + 1)  # 解答なしの迷路を保存
    draw_maze(maze_grid, i + 1, solution_path=solution_path, is_solution=True)  # 解答付き迷路を保存

print(f"{num_mazes} 個の迷路とその解答がそれぞれのディレクトリに保存されました。")
```

</details>

## 学習方法
今回は、 **GAN** を用いて画像生成をした。GANには **生成器** と **判別器** の2種類が存在する。
- 生成器 ... 入力画像をもとに画像を生成し出力するモデル。
- 判別器 ... 生成器が出力した画像と用意した正解画像を見分けるモデル。

生成器と判別器についてどのようなものを扱っているか、どのような処理が行われているかを述べる。

### 生成器
生成器は **U-Net** である。U-Netは、 **エンコーダ** と **デコーダ** の2つの部分から構成され、画像の特徴を抽出した後に画像を再構築するネットワークである。

エンコーダは入力画像を圧縮し、特徴を抽出する役割を持つ。一方で、デコーダは圧縮された特徴をもとに画像を再構築する。エンコーダの各層とデコーダの各層が対応しており、それらの層は直接接続されている。これにより詳細な情報が伝達され、精度の高い出力画像が得られる。

### 判別器
判別器は **PatchGAN Discriminator** である。入力画像(サイズ128x128)を畳み込んでできた特徴マップ(サイズ7x7)を作り、各パッチ(計7x7=49個)ごとに生成された画像が本物なのか偽物なのかを示すスコアを出力する。

この時、入力画像は2チャンネルの画像である。カラー画像がR, G, Bの3チャンネル画像であるのと同じように **1チャンネル目に迷路画像, 2チャンネル目に正解ルート描写画像or生成器が出力した画像** とすることで、2チャンネルの画像にして判別器に入力する。

<br>

このように生成器, 判別器の二つのモデルを学習する。学習方法としては下記の通りである。

1. 生成器が正解ルート描写画像を生成
2. 生成した画像を判別器に入力し、本物かどうかを判断
3. バイナリ分類におけるクロスエントロピー損失(生成した画像が **本物** であるとした時の損失)と生成した画像と正解画像のピクセルごとの差の絶対値を計算して求められる損失を合計した損失から生成器を更新
4. 迷路画像と正解画像のペア, 迷路画像と生成画像のペアをそれぞれ判別器に入力
5. バイナリ分類におけるクロスエントロピー損失(正解画像でのペアは **本物** 、生成画像でのペアは **偽物** とした時の損失)から判別器を更新。

このように、生成器は判別器を騙せるような画像を生成できるように学習し、判別器は生成画像を本物と間違わないように学習する。このように学習を進めることで、 **クオリティの高い判別器を騙せる生成器を作ることができる。**

<details>
<summary>生成器定義コード</summary>

```python
# 生成器

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, features=64):
        super(UNetGenerator, self).__init__()

        # エンコーダ部分

        self.down1 = self.conv_block(input_channels, features, normalize=False)  # 1 -> 64
        self.down2 = self.conv_block(features, features * 2)                     # 64 -> 128
        self.down3 = self.conv_block(features * 2, features * 4)                 # 128 -> 256
        self.down4 = self.conv_block(features * 4, features * 8)                 # 256 -> 512

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, kernel_size=4, stride=2, padding=1),  # 512 -> 1024
            nn.ReLU(),
        )

        # デコーダ部分
        self.up1 = self.upconv_block(features * 16, features * 8)                # 1024 -> 512
        self.up2 = self.upconv_block(features * 8 * 2, features * 4)             # (512 + 512) -> 256
        self.up3 = self.upconv_block(features * 4 * 2, features * 2)             # (256 + 256) -> 128
        self.up4 = self.upconv_block(features * 2 * 2, features)                 # (128 + 128) -> 64

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(features * 2, output_channels, kernel_size=4, stride=2, padding=1),  # (64 + 64) -> 1
            nn.Tanh(),
        )

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # エンコーダ
        e1 = self.down1(x)  # 1 -> 64
        e2 = self.down2(e1)  # 64 -> 128
        e3 = self.down3(e2)  # 128 -> 256
        e4 = self.down4(e3)  # 256 -> 512

        # ボトルネック
        bn = self.bottleneck(e4)  # 512 -> 1024

        # デコーダ
        d1 = self.up1(bn)                       # 1024 -> 512
        d1 = torch.cat([d1, e4], dim=1)         # 512 + 512 = 1024

        d2 = self.up2(d1)                       # 1024 -> 256
        d2 = torch.cat([d2, e3], dim=1)         # 256 + 256 = 512

        d3 = self.up3(d2)                       # 512 -> 128
        d3 = torch.cat([d3, e2], dim=1)         # 128 + 128 = 256

        d4 = self.up4(d3)                       # 256 -> 64
        d4 = torch.cat([d4, e1], dim=1)         # 64 + 64 = 128

        output = self.final_layer(d4)           # 128 -> 1

        return output
```

</details>

<details>
<summary>判別器定義コード</summary>

```python
# 判別器

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=2, features=64):  # 1 + 1 = 2
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self.disc_block(input_channels, features, normalize=False),
            self.disc_block(features, features * 2),
            self.disc_block(features * 2, features * 4),
            self.disc_block(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def disc_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        img_input = torch.cat([x, y], dim=1)
        return self.model(img_input)
```

</details>

## 結果
まずは学習時の損失を紹介する。以下は生成器の損失(学習時と検証時)と判別器の損失である(Loss_G ... 学習データでの生成器の損失。Loss_D ... 学習データでの判別器の損失。Val_Loss ... 検証用データでの生成器のL1損失)。

(なお、本当はこれの前に数十エポックは行われているはずだが、colab notebookのバグなのか、最初の20分以上の学習を出力してくれていない？)

Epoch [1/100] Loss_G: 21.0084 Loss_D: 0.0010 Val_Loss: 13.5319

Epoch [2/100] Loss_G: 19.6205 Loss_D: 0.0006 Val_Loss: 11.7064

Epoch [3/100] Loss_G: 18.7184 Loss_D: 0.0004 Val_Loss: 11.3023

Epoch [4/100] Loss_G: 5.8163 Loss_D: 0.6613 Val_Loss: 3.0838

Epoch [5/100] Loss_G: 2.4343 Loss_D: 0.6886 Val_Loss: 2.5419

Epoch [6/100] Loss_G: 1.9796 Loss_D: 0.6926 Val_Loss: 2.4617

Epoch [7/100] Loss_G: 1.7180 Loss_D: 0.6928 Val_Loss: 2.1081

Epoch [8/100] Loss_G: 1.5679 Loss_D: 0.6922 Val_Loss: 1.9957

Epoch [9/100] Loss_G: 1.4606 Loss_D: 0.6924 Val_Loss: 1.9666

初めは生成器も判別器も損失が下がっていたが、先に判別器の損失がほぼ0になった(クオリティの高い判別器が完成)。その判別器を騙すような画像を生成できるように生成器は学習する。

下記の画像は生成器は騙すには未熟だった時の画像である。

(生成器甘.png)

スタート部分とゴール部分からルートが塗られているのがわかるが、余計な黒点が大量に出てしまっている。

しかし、急に生成器の損失が下がり、判別器の損失が上がった。これは、 **生成器が本物に近い画像を生成することができるようになり、判別器が正しく判断できなくなってしまった** ということである。

100エポックまでやった後、検証用データ使用時の生成器の損失が一番低かった時のモデルを使用して、新規データ50件で結果を確認する(なお、100エポック目のVal_lossが一番低かったのでその時のモデルを使用)。

以下はうまくいった例である。

(成功例_1.png)
(成功例_2.png)

一方で以下のようにうまくいってない例もある。

無駄に道を塗っている生成画像例：

(過検出_1.png)
(過検出_2.png)

ルートが途中で途切れている生成画像例：

(未検出_1.png)
(未検出_2.png)

50件の割り振りは以下のようになった。

| 成功 | ルートの過検出 | ルートの未検出 |
|---|---|---|
| 25件 | 11件 | 14件 |

9x9の迷路盤面のパターンは膨大なので、5000件(学習で使われるのはさらに少ない4000件)だと少なすぎるというのが原因と考えられる。より精度の良い生成器を作るならばデータセットを増やした方がいい(ただし、無課金でgoogleのcolab notebookを使用しているため、GPUの使用時間に上限があり、データ数を増やすのを躊躇っていた。)

## まとめ

まだまだミスはあるものの、データ数5000件(そのうち学習に使用したのは4000件)という少ないデータ数で学習にかかった時間が約1.5時間と短いわりには、比較的良い確率で正解ルートを描写してもらうことに成功した。

もし、今後さらにこの自己学習を行うのならば、以下のことを試したいとも思っている

- さらに大きい盤面での学習、それに伴って画像サイズも大きくして、モデルの層も増やす。
- attention機構が使えないか試す(通らない道を考慮して欲しくないので、attention機構で重要なとこにだけ注意してもらったらより精度が良くなるのではと思った)。

## コード
googleのcolab notebook上でT4 GPUを使用して行った。

```python
from google.colab import drive
drive.mount('/content/drive')

# 画像認識・生成による迷路

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
```

```python
# データセットのクラス定義
class MazeDataset(Dataset):
    def __init__(self, maze_dir, solution_dir, transform=None):
        self.maze_dir = maze_dir
        self.solution_dir = solution_dir
        self.transform = transform
        # 拡張子が.pngのファイルのみを取得
        self.maze_files = sorted([f for f in os.listdir(maze_dir) if f.endswith('.png')])
        self.solution_files = sorted([f for f in os.listdir(solution_dir) if f.endswith('.png')])

    def __len__(self):
        return min(len(self.maze_files), len(self.solution_files))

    def __getitem__(self, idx):
        maze_path = os.path.join(self.maze_dir, self.maze_files[idx])
        solution_path = os.path.join(self.solution_dir, self.solution_files[idx])

        maze_image = Image.open(maze_path).convert('L')
        solution_image = Image.open(solution_path).convert('L')

        if self.transform:
            maze_image = self.transform(maze_image)
            solution_image = self.transform(solution_image)

        return maze_image, solution_image
```

```python
# 変換とデータローダーの作成
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]に正規化
])

# データセットの読み込み
dataset = MazeDataset('/content/drive/MyDrive/mazes/maze', '/content/drive/MyDrive/mazes/solution', transform=transform)

# データセットの分割
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

```python
# 生成器

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, features=64):
        super(UNetGenerator, self).__init__()

        # エンコーダ部分

        self.down1 = self.conv_block(input_channels, features, normalize=False)  # 1 -> 64
        self.down2 = self.conv_block(features, features * 2)                     # 64 -> 128
        self.down3 = self.conv_block(features * 2, features * 4)                 # 128 -> 256
        self.down4 = self.conv_block(features * 4, features * 8)                 # 256 -> 512

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, kernel_size=4, stride=2, padding=1),  # 512 -> 1024
            nn.ReLU(),
        )

        # デコーダ部分
        self.up1 = self.upconv_block(features * 16, features * 8)                # 1024 -> 512
        self.up2 = self.upconv_block(features * 8 * 2, features * 4)             # (512 + 512) -> 256
        self.up3 = self.upconv_block(features * 4 * 2, features * 2)             # (256 + 256) -> 128
        self.up4 = self.upconv_block(features * 2 * 2, features)                 # (128 + 128) -> 64

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(features * 2, output_channels, kernel_size=4, stride=2, padding=1),  # (64 + 64) -> 1
            nn.Tanh(),
        )

    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # エンコーダ
        e1 = self.down1(x)  # 1 -> 64
        e2 = self.down2(e1)  # 64 -> 128
        e3 = self.down3(e2)  # 128 -> 256
        e4 = self.down4(e3)  # 256 -> 512

        # ボトルネック
        bn = self.bottleneck(e4)  # 512 -> 1024

        # デコーダ
        d1 = self.up1(bn)                       # 1024 -> 512
        d1 = torch.cat([d1, e4], dim=1)         # 512 + 512 = 1024

        d2 = self.up2(d1)                       # 1024 -> 256
        d2 = torch.cat([d2, e3], dim=1)         # 256 + 256 = 512

        d3 = self.up3(d2)                       # 512 -> 128
        d3 = torch.cat([d3, e2], dim=1)         # 128 + 128 = 256

        d4 = self.up4(d3)                       # 256 -> 64
        d4 = torch.cat([d4, e1], dim=1)         # 64 + 64 = 128

        output = self.final_layer(d4)           # 128 -> 1

        return output
```

```python
# 判別器

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=2, features=64):  # 1 + 1 = 2
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self.disc_block(input_channels, features, normalize=False),
            self.disc_block(features, features * 2),
            self.disc_block(features * 2, features * 4),
            self.disc_block(features * 4, features * 8),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def disc_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        img_input = torch.cat([x, y], dim=1)
        return self.model(img_input)
```

```python
# 損失関数と最適化の定義

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
```

```python
# 学習

num_epochs = 100
best_val_loss = float('inf')  # 最良の検証損失を保持

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    # 学習ループ
    for i, (maze, solution) in enumerate(train_loader):
        maze = maze.to(device)
        solution = solution.to(device)

        ## 1. 生成器の更新
        optimizer_G.zero_grad()

        # 生成器が生成した画像
        fake_solution = generator(maze)

        # 判別器を使用して偽物の画像を判定
        pred_fake = discriminator(maze, fake_solution)

        # 判別器の出力サイズに合わせてラベルを作成
        real_label = torch.ones_like(pred_fake, device=device)

        # 生成器の損失
        loss_GAN = criterion_GAN(pred_fake, real_label)
        loss_L1 = criterion_L1(fake_solution, solution) * 100
        loss_G = loss_GAN + loss_L1

        loss_G.backward()
        optimizer_G.step()

        ## 2. 判別器の更新
        optimizer_D.zero_grad()

        # 本物の組み合わせ
        pred_real = discriminator(maze, solution)
        real_label = torch.ones_like(pred_real, device=device)
        loss_D_real = criterion_GAN(pred_real, real_label)

        # 偽物の組み合わせ
        pred_fake = discriminator(maze, fake_solution.detach())
        fake_label = torch.zeros_like(pred_fake, device=device)
        loss_D_fake = criterion_GAN(pred_fake, fake_label)

        # 判別器の損失
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        optimizer_D.step()

        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()

    # 検証ループ
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        for maze, solution in val_loader:
            maze = maze.to(device)
            solution = solution.to(device)

            fake_solution = generator(maze)
            loss_L1 = criterion_L1(fake_solution, solution) * 100
            val_loss += loss_L1.item()

    val_loss /= len(val_loader)

    # 最良モデルの保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(generator.state_dict(), 'best_generator.pth')

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss_G: {running_loss_G/len(train_loader):.4f} "
          f"Loss_D: {running_loss_D/len(train_loader):.4f} "
          f"Val_Loss: {val_loss:.4f}")

    # 定期的に生成画像チェック
    if (epoch + 1) % 10 == 0:
        generator.eval()
        with torch.no_grad():
            sample_maze = maze[0:1]  # バッチの最初の迷路を取得
            sample_solution = solution[0:1]  # 対応する解法画像
            generated_solution = generator(sample_maze)

            # 出力を[0, 1]の範囲に変換
            generated_solution = (generated_solution + 1) / 2
            generated_solution = generated_solution.clamp(0, 1)

            # 画像として保存
            transforms.ToPILImage()(generated_solution.squeeze().cpu()).save(f"generated_sample_epoch_{epoch+1}.png")

# 最終的なモデルの保存
torch.save(generator.state_dict(), 'generator.pth')
```

```python
# 推論

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化と重みのロード（最良モデルを使用）
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load('best_generator.pth', map_location=device))
generator.eval()

# 推論用のデータセットの準備
class MazeTestDataset(Dataset):
    def __init__(self, maze_dir, transform=None):
        self.maze_dir = maze_dir
        self.transform = transform
        # 拡張子が.pngのファイルのみを取得
        self.maze_files = sorted([f for f in os.listdir(maze_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.maze_files)

    def __getitem__(self, idx):
        maze_path = os.path.join(self.maze_dir, self.maze_files[idx])
        maze_image = Image.open(maze_path).convert('L')  # グレースケールに変更

        if self.transform:
            maze_image = self.transform(maze_image)

        return maze_image, self.maze_files[idx]  # ファイル名も返す

# 変換の定義（トレーニング時と同じ）
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]に正規化
])

# テスト用データセットとデータローダーの作成
test_dataset = MazeTestDataset('/content/drive/MyDrive/mazes/maze_test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 出力先ディレクトリの作成
output_dir = 'generated_solutions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 推論と結果の保存
with torch.no_grad():
    for maze_image, filename in test_loader:
        maze_image = maze_image.to(device)

        # 生成器による解画像の生成
        fake_solution = generator(maze_image)

        # 出力を[0, 1]の範囲に変換
        fake_solution = (fake_solution + 1) / 2  # Tanhの出力を0〜1にスケーリング

        # ピクセル値をクリップして範囲外の値を補正
        fake_solution = fake_solution.clamp(0, 1)

        # Tensorを画像に変換
        fake_solution_image = transforms.ToPILImage()(fake_solution.squeeze().cpu())

        # filenameから文字列を抽出
        filename_str = filename[0] if isinstance(filename, (list, tuple)) else filename

        # 画像の保存
        fake_solution_image.save(os.path.join(output_dir, f'solution_{filename_str}'))
```

## 余談
実は9x9盤面でやるよりも前に7x7盤面でも行っていた。

7x7盤面での50件の割り振りは以下のようになった。

| 成功 | ルートの過検出 | ルートの未検出 |
|---|---|---|
| 40件 | 9件 | 1件 |

その時のlossは正常に残っていたので共有する。判別器が精度良くなってからかなり時間が経って生成器の精度が良くなったのが良くわかる。

<details>
<summary>各種のロス</summary>

Epoch [1/100] Loss_G: 12.8379 Loss_D: 0.0035 Val_Loss: 5.9550

Epoch [2/100] Loss_G: 11.5746 Loss_D: 0.0021 Val_Loss: 4.5727

Epoch [3/100] Loss_G: 11.5038 Loss_D: 0.0008 Val_Loss: 4.2537

Epoch [4/100] Loss_G: 11.6091 Loss_D: 0.0005 Val_Loss: 4.0706

Epoch [5/100] Loss_G: 11.6908 Loss_D: 0.0004 Val_Loss: 3.7302

Epoch [6/100] Loss_G: 11.7844 Loss_D: 0.0003 Val_Loss: 3.6192

Epoch [7/100] Loss_G: 11.8544 Loss_D: 0.0002 Val_Loss: 3.5244

Epoch [8/100] Loss_G: 11.9513 Loss_D: 0.0002 Val_Loss: 3.3685

Epoch [9/100] Loss_G: 12.0293 Loss_D: 0.0002 Val_Loss: 3.3438

Epoch [10/100] Loss_G: 12.1238 Loss_D: 0.0001 Val_Loss: 3.2189

Epoch [11/100] Loss_G: 4.6921 Loss_D: 0.5754 Val_Loss: 1.7281

Epoch [12/100] Loss_G: 1.5797 Loss_D: 0.6942 Val_Loss: 1.2411

Epoch [13/100] Loss_G: 1.4212 Loss_D: 0.6938 Val_Loss: 1.1627

Epoch [14/100] Loss_G: 1.3440 Loss_D: 0.6930 Val_Loss: 1.0452

Epoch [15/100] Loss_G: 1.3019 Loss_D: 0.6928 Val_Loss: 1.1243

Epoch [16/100] Loss_G: 1.2636 Loss_D: 0.6927 Val_Loss: 1.0429

Epoch [17/100] Loss_G: 1.2373 Loss_D: 0.6926 Val_Loss: 0.9652

Epoch [18/100] Loss_G: 1.2106 Loss_D: 0.6927 Val_Loss: 1.0250

Epoch [19/100] Loss_G: 1.1966 Loss_D: 0.6925 Val_Loss: 0.9713

Epoch [20/100] Loss_G: 1.1793 Loss_D: 0.6924 Val_Loss: 0.9084

Epoch [21/100] Loss_G: 1.1638 Loss_D: 0.6921 Val_Loss: 0.9820

Epoch [22/100] Loss_G: 1.1617 Loss_D: 0.6933 Val_Loss: 0.9445

Epoch [23/100] Loss_G: 1.1371 Loss_D: 0.6928 Val_Loss: 0.9281

Epoch [24/100] Loss_G: 1.1281 Loss_D: 0.6929 Val_Loss: 0.9014

Epoch [25/100] Loss_G: 1.1253 Loss_D: 0.6919 Val_Loss: 0.8944

Epoch [26/100] Loss_G: 1.1108 Loss_D: 0.6921 Val_Loss: 0.8770

Epoch [27/100] Loss_G: 1.1109 Loss_D: 0.6917 Val_Loss: 0.9262

Epoch [28/100] Loss_G: 1.1008 Loss_D: 0.6903 Val_Loss: 0.9140

Epoch [29/100] Loss_G: 1.1072 Loss_D: 0.6900 Val_Loss: 0.8801

Epoch [30/100] Loss_G: 1.0919 Loss_D: 0.6936 Val_Loss: 0.8744

Epoch [31/100] Loss_G: 1.0989 Loss_D: 0.6872 Val_Loss: 0.9135

</details>
