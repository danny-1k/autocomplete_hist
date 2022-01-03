window.onload = ()=>{
    document.querySelector('#input').oninput=()=>{
        var text = document.querySelector('#input').value;
        document.querySelector('.suggestions').innerHTML = '';

        if (text !==''){

            var xhr = new XMLHttpRequest();
            xhr.open('POST','http://localhost:5000/suggest',true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.onload = (e)=>{
                if(xhr.readyState === 4){
                    if(xhr.status === 200){
                        document.querySelector('.suggestions').innerHTML = '';

                        var suggestions = JSON.parse(xhr.responseText)['suggestions'];
                        for(var suggestion of suggestions){
                            if(suggestion !== 'UNK'){
                                var sugg = document.createElement('div');
                                sugg.classList.add('suggestion');
                                sugg.innerHTML = document.querySelector('#input').value+' '+suggestion;
                                document.querySelector('.suggestions').appendChild(sugg);

                            };

                        };
                    };
                };
            };
            xhr.send(JSON.stringify({text:text}));
        };
    };
};