---
layout: default
---
<p style="font-style: italic; text-align: right; font-size: 90%"><a href="/jekyll/categories">Listings by categories &rarr;</a></p>

{% for post in site.posts %}

  {% assign currentdate = post.date | date: "%B %Y" %}
  {% if currentdate != date %}
  <h4 id="y{{post.date | date: "%Y"}}"> {{ currentdate }} </h4>
  <hr/>
    {% assign date = currentdate %}
  {% endif %}

&bull; {{ post.date | date: "%d" }}&nbsp; - &nbsp;<a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a> &nbsp; - &nbsp; {% if post.categories %} <small>categories: <em>{{ post.categories | join: "</em> - <em>" }}</em></small> {% endif %} {::comment} {% if post.tags %} <small>, tags: <em>{{ post.tags | join: "</em> - <em>" }}</em></small> {% endif %}{:/comment}

{% endfor %}
