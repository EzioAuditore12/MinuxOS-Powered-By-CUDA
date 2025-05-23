org 0x0
bits 16

%define ENDL 0x0D, 0x0A

start:
    ; print message
    mov si, msg_hello
    call puts           ; use 'call', not 'tail'

.halt:
    cli
    hlt

; Prints a string on the screen
; Params:
;       - ds:si points to string
puts:
    ; save registers we will modify
    push si
    push ax

.loop:
    lodsb               ; load next character into AL
    or al, al           ; check if null terminator
    jz .done

    mov ah, 0x0E        ; teletype BIOS service
    int 0x10

    jmp .loop

.done:
    pop ax
    pop si
    ret

msg_hello: db 'Welcome here!', ENDL, 0

times 510 - ($ - $$) db 0
dw 0xAA55

