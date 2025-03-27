$(document).ready(function() {
    $('.chat-button').on('click', function() {
        $('.chat-button').hide();
        $('.chat-box').css("visibility", "visible");
    });

    $('.chat-box-header p').on('click', function() {
        $('.chat-button').show();
        $('.chat-box').css("visibility", "hidden");
    });

    function addMessage(message, isReceived) {
        const messageDiv = $(`<div class="chat-box-body-${isReceived ? 'receive' : 'send'}">`);
        messageDiv.html(`
            <p>${message}</p>
            <span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        `);
        $('#chat-messages').append(messageDiv);
        $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
    }

    function sendMessage() {
        const message = $('#message-input').val().trim();
        if (!message) return;

        addMessage(message, false);
        $('#message-input').val('');

        $.ajax({
            url: '/send_message',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(response) {
                addMessage(response.message, true);
            }
        });
    }

    $('#send-button').click(sendMessage);
    $('#message-input').keypress(function(e) {
        if (e.which == 13) sendMessage();
    });
});