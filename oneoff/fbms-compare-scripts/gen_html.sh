grep Sequence ./comparison.log | sed -e 's/.*Sequence: \(.*\)Numbers.txt,.*/\1/g' | while read line ; do
echo "<h3>${line}</h3>" >> videos.html
echo "<video style='width: 1000px;' controls src='${line}.mp4'></video>" >> videos.html
done
