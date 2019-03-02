---
layout: default
---

# Jekyll cheat sheet

----
```
Text can be **bold**, _italic_, or ~~strikethrough~~.
```
Text can be **bold**, _italic_, or ~~strikethrough~~.

----
```
There should be whitespace between paragraphs.

There should be whitespace between paragraphs.
```
There should be whitespace between paragraphs.

There should be whitespace between paragraphs.

----
```
# Header 1
```
# Header 1

This is a normal paragraph following a header.

----
```
## Header 2
> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.
```
## Header 2
> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

----
```
### Header 3
```
### Header 3

----
    ```python
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(2, input_dim=2, use_bias=True))  # hidden layer
    model.add(Activation('tanh'))
    model.add(Dense(1, use_bias=True))               # output layer
    model.add(Activation('sigmoid'))
    ```

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True))  # hidden layer
model.add(Activation('tanh'))
model.add(Dense(1, use_bias=True))               # output layer
model.add(Activation('sigmoid'))
```

----
    ```js
    // Javascript code with syntax highlighting.
    var fun = function lang(l) {
      dateformat.i18n = require('./lang/' + l)
      return true;
    }
    ```

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

----
    ```ruby
    # Ruby code with syntax highlighting
    GitHubPages::Dependencies.gems.each do |gem, version|
      s.add_dependency(gem, "= #{version}")
    end
    ```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

----
```
#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

----
```
##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.
```
##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

----
```
###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |
```
###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

----
```
### There's a horizontal rule below this.

* * *
```
### There's a horizontal rule below this.

* * *

----
```
### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip
```
### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

----
```
### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four
```
### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

----
```
### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item
```
### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

----
```
### Small image
<svg width="22" height="22" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
```
### Small image
<svg width="22" height="22" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>

----
```
### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)
```
### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


----
```
### Definition lists

Name
: Godzilla

Born
: 1952

Birthplace
: Japan

Color
: Green
```
### Definition lists

Name
: Godzilla

Born
: 1952

Birthplace
: Japan

Color
: Green

----
    ```
    Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
    ```

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

----
```
$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$

$$k_{n+1} = n^2 + k_n^2 - k_{n-1}$$
```

$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$

$$k_{n+1} = n^2 + k_n^2 - k_{n-1}$$

----
漢字
ひらがな
カタカナ

----
```
The final element.
```
