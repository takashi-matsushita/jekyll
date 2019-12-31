git clone ssh://git@github.com/takashi-matsushita/jekyll
cd jekyll
git fetch
git checkout gh-pages
bundle update
bundle exec jekyll serve
