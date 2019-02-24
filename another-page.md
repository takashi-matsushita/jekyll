---
layout: post
title: blogs
---

## Welcome to another page

_yay_


## Blog Posts

{% for post in site.posts %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}


[back](./)
